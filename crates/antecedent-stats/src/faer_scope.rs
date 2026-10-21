//! Per-operation worker limit and scratch budget for `faer`.
//!
//! A [`FaerScope`] carries the caller's worker limit. Scratch bytes come from
//! an [`AdmissionLease`](antecedent_core::execution::AdmissionLease), the same account as
//! the rest of a study. [`FaerScope::parallelism`] is the degree a kernel passes into `faer`:
//! [`faer::Par::Seq`] for one worker, and [`faer::Par::Rayon`] with that exact
//! count when the limit is larger. The count is never zero. This module does
//! not call [`faer::Par::rayon`], [`faer::get_global_parallelism`], or
//! [`faer::set_global_parallelism`]. `Par::rayon(0)` reads the process-global
//! rayon pool's current width. That operation is not provided, and neither are
//! the convenience factorizations that read the same pool (`ColPivQr::new`,
//! `thin_svd`, `self_adjoint_eigenvalues`, `self_adjoint_eigen`).
//!
//! [`FaerScope::admit_scratch`] checks a [`faer::dyn_stack::StackReq`] against
//! the lease remainder and draws a child lease before [`FaerScope::scratch_buffer`]
//! allocates. A requirement
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

use antecedent_core::execution::AdmissionLease;
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
}

/// A faer scratch buffer and the child lease that pays for it.
pub struct FaerScratch {
    /// Bytes the kernel reads. The lease stays alive until this buffer is dropped.
    pub buffer: MemBuffer,
    /// Child lease funding `buffer`. Read by drop, so the bytes return to the pool.
    #[allow(dead_code)]
    lease: AdmissionLease,
}

impl FaerScope {
    /// Scope with `workers`. Scratch bytes are drawn from an [`AdmissionLease`].
    ///
    /// `workers` must be at least one. This does not spawn threads and does not
    /// read the process pool.
    ///
    /// # Errors
    ///
    /// [`FaerScopeError::ZeroWorkers`] when `workers` is zero.
    #[must_use = "a refused scope does not carry a worker limit"]
    pub const fn try_new(workers: u32) -> Result<Self, FaerScopeError> {
        match NonZeroU32::new(workers) {
            Some(workers) => Ok(Self { workers }),
            None => Err(FaerScopeError::ZeroWorkers),
        }
    }

    /// Worker limit declared for this scope. At least one.
    #[must_use]
    pub const fn workers(self) -> u32 {
        self.workers.get()
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

    /// Draw a child lease for `required` from `parent`.
    ///
    /// A layout `faer` cannot satisfy is [`FaerScopeError::ScratchOverflow`].
    /// A requirement above the parent remainder is [`FaerScopeError::ScratchOverBudget`].
    /// Neither path allocates a buffer. A refusal leaves `parent` unchanged.
    ///
    /// # Errors
    ///
    /// [`FaerScopeError::ScratchOverflow`] when `required` has no layout.
    /// [`FaerScopeError::ScratchOverBudget`] when the layout's byte size is
    /// greater than `parent`'s remaining bytes.
    #[must_use = "a scratch refusal means the kernel must not allocate that buffer"]
    pub fn admit_scratch(
        self,
        parent: &AdmissionLease,
        required: StackReq,
    ) -> Result<AdmissionLease, FaerScopeError> {
        let required_bytes = scratch_bytes(required)?;
        let budget_bytes = parent.remainder().bytes();
        if required_bytes > budget_bytes {
            return Err(FaerScopeError::ScratchOverBudget { required_bytes, budget_bytes });
        }
        parent
            .acquire_child(0, required_bytes)
            .map_err(|_| FaerScopeError::ScratchOverBudget { required_bytes, budget_bytes })
    }

    /// Allocate `required` after [`Self::admit_scratch`] accepts it.
    ///
    /// The returned scratch holds the child lease until it is dropped.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::admit_scratch`]. A refusal allocates nothing.
    #[must_use = "a scratch refusal means the kernel must not allocate that buffer"]
    pub fn scratch_buffer(
        self,
        parent: &AdmissionLease,
        required: StackReq,
    ) -> Result<FaerScratch, FaerScopeError> {
        let lease = self.admit_scratch(parent, required)?;
        Ok(FaerScratch { buffer: MemBuffer::new(required), lease })
    }
}

impl FaerScratch {
    /// The buffer `faer` writes into.
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut MemBuffer {
        &mut self.buffer
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
    use antecedent_core::{ExecutionContext, execution::AdmissionPool};
    use std::num::NonZeroUsize;

    use faer::Par;
    use faer::dyn_stack::StackReq;

    use antecedent_core::execution::AdmissionPool;

    use super::{FaerScope, FaerScopeError};

    fn lease_bytes(bytes: u64) -> antecedent_core::execution::AdmissionLease {
        let pool = AdmissionPool::fixed(1, bytes);
        ExecutionContext::for_tests(1)
            .with_admission(pool)
            .acquire_admission(0, bytes)
            .expect("the pool funds this lease")
    }

    #[test]
    fn two_scopes_declare_different_worker_limits() {
        let narrow = FaerScope::try_new(1).unwrap();
        let wide = FaerScope::try_new(4).unwrap();
        assert_eq!(narrow.workers(), 1);
        assert_eq!(wide.workers(), 4);
        assert_ne!(narrow.workers(), wide.workers());
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
        assert_eq!(FaerScope::try_new(0), Err(FaerScopeError::ZeroWorkers));
        let held = FaerScope::try_new(2).unwrap();
        assert_eq!(held.workers(), 2);
        assert_eq!(FaerScope::try_new(0), Err(FaerScopeError::ZeroWorkers));
    }

    #[test]
    fn scratch_above_the_lease_is_refused() {
        let scope = FaerScope::try_new(1).unwrap();
        let parent = lease_bytes(8);
        let eight = StackReq::new::<u8>(8);
        let child = scope.admit_scratch(&parent, eight).unwrap();
        assert_eq!(child.remainder().bytes(), 8);
        assert_eq!(parent.remainder().bytes(), 0);
        drop(child);
        assert_eq!(parent.remainder().bytes(), 8);
        assert_eq!(
            scope.admit_scratch(&parent, StackReq::new::<u8>(9)).unwrap_err(),
            FaerScopeError::ScratchOverBudget { required_bytes: 9, budget_bytes: 8 }
        );
        assert_eq!(parent.remainder().bytes(), 8);
        assert_eq!(
            scope.admit_scratch(&parent, StackReq::OVERFLOW).unwrap_err(),
            FaerScopeError::ScratchOverflow
        );
        let buffer = scope.scratch_buffer(&parent, eight).unwrap();
        assert!(buffer.buffer.len() >= 8);
        assert_eq!(parent.remainder().bytes(), 0);

        let empty = lease_bytes(0);
        let empty_scope = FaerScope::try_new(2).unwrap();
        let zero = empty_scope.admit_scratch(&empty, StackReq::empty()).unwrap();
        assert_eq!(zero.remainder().bytes(), 0);
        drop(zero);
        assert_eq!(
            empty_scope.admit_scratch(&empty, StackReq::new::<u8>(1)).unwrap_err(),
            FaerScopeError::ScratchOverBudget { required_bytes: 1, budget_bytes: 0 }
        );
        assert_eq!(empty_scope.workers(), 2);
        assert_ne!(empty_scope.workers(), scope.workers());
    }

    #[test]
    fn scratch_buffer_above_the_lease_is_refused() {
        let scope = FaerScope::try_new(3).unwrap();
        let parent = lease_bytes(4);
        assert_eq!(scope.workers(), 3);
        assert_eq!(scope.parallelism().degree(), 3);
        assert!(scope.parallelism().degree() <= usize::try_from(scope.workers()).unwrap());
        let Err(error) = scope.scratch_buffer(&parent, StackReq::new::<u8>(5)) else {
            panic!("a scratch request above the lease must be refused");
        };
        assert_eq!(error, FaerScopeError::ScratchOverBudget { required_bytes: 5, budget_bytes: 4 });
        assert_eq!(parent.remainder().bytes(), 4);
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
