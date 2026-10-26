// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carry a public workflow's admission authority through context-free numerical traits.
//! This is scoped thread state, never shared mutable prepared state or a second budget.
use crate::{
    ExecutionContext,
    execution::{AdmissionError, AdmissionLease},
};
use std::{
    cell::RefCell,
    sync::{Arc, Mutex, PoisonError},
};

#[derive(Clone, Default)]
struct Binding {
    context: Option<ExecutionContext>,
    reservation: Option<Arc<AdmissionLease>>,
    worker: Option<Arc<AdmissionLease>>,
    failure: Option<Arc<Mutex<Option<AdmissionError>>>>,
}
thread_local! { static ACTIVE: RefCell<Binding> = RefCell::default(); }
struct Restore(Binding);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = std::mem::take(&mut self.0));
    }
}
/// Worker-carried authority for the same pool and any already admitted workspace.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct NumericalAdmissionToken(Binding);
impl NumericalAdmissionToken {
    /// Capture the current workflow binding.
    #[must_use]
    pub fn capture() -> Self {
        Self(ACTIVE.with(|active| active.borrow().clone()))
    }
    /// Install this binding on a worker, using its bounded inner context.
    pub fn execute<T>(&self, ctx: &ExecutionContext, execute: impl FnOnce() -> T) -> T {
        let binding = Binding {
            context: Some(ctx.clone()),
            reservation: self.0.reservation.clone(),
            worker: self.0.worker.clone(),
            failure: self.0.failure.clone(),
        };
        let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
        execute()
    }
}
/// Install the caller's execution policy while preserving an enclosing workspace reservation.
#[doc(hidden)]
pub fn with_context<T>(ctx: &ExecutionContext, execute: impl FnOnce() -> T) -> T {
    NumericalAdmissionToken::capture().execute(ctx, execute)
}

/// Preserve the first actual numerical admission failure when a component's
/// legacy error conversion erases its operational type. Nested scopes own
/// independent records, and worker tokens carry the current record explicitly.
#[doc(hidden)]
pub fn with_context_and_failure<T>(
    ctx: &ExecutionContext,
    execute: impl FnOnce() -> T,
) -> (T, Option<AdmissionError>) {
    let failure = Arc::new(Mutex::new(None));
    let mut binding = ACTIVE.with(|active| active.borrow().clone());
    binding.context = Some(ctx.clone());
    binding.failure = Some(Arc::clone(&failure));
    let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
    let result = execute();
    let first = *failure.lock().unwrap_or_else(PoisonError::into_inner);
    (result, first)
}
/// Make the already admitted compiled workspace available to original numerical kernels.
#[doc(hidden)]
pub fn with_reservation<T>(reservation: Arc<AdmissionLease>, execute: impl FnOnce() -> T) -> T {
    let mut binding = ACTIVE.with(|active| active.borrow().clone());
    binding.reservation = Some(reservation);
    let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
    execute()
}
/// Draw actual scratch bytes from the current workflow. Context-free component calls return None.
#[doc(hidden)]
#[must_use]
pub fn acquire(bytes: u64) -> Option<Result<AdmissionLease, AdmissionError>> {
    let binding = ACTIVE.with(|active| active.borrow().clone());
    let result = if let Some(reservation) = binding.reservation {
        Some(reservation.acquire_child(0, bytes))
    } else {
        binding.context.map(|context| context.acquire_admission(0, bytes))
    };
    if let (Some(Err(error)), Some(failure)) = (&result, binding.failure) {
        failure.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(*error);
    }
    result
}

/// Check the active workflow's cancellation at an original numerical loop boundary.
#[doc(hidden)]
#[must_use]
pub fn is_cancelled() -> bool {
    ACTIVE.with(|active| {
        active.borrow().context.as_ref().is_some_and(|ctx| ctx.cancellation.is_cancelled())
    })
}

/// Current worker reservation, retained only for the dynamic worker scope.
pub(crate) fn worker() -> Option<Arc<AdmissionLease>> {
    ACTIVE.with(|active| active.borrow().worker.clone())
}
/// Install the real worker lease so nested admitted maps reuse that CPU slot.
pub(crate) fn with_worker<T>(worker: Arc<AdmissionLease>, execute: impl FnOnce() -> T) -> T {
    let mut binding = ACTIVE.with(|active| active.borrow().clone());
    binding.worker = Some(worker);
    let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
    execute()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_failure_capture_is_first_only_and_nested_scopes_restore_the_parent() {
        let outer = ExecutionContext::for_tests(1).with_memory_limit(Some(8));
        let inner = ExecutionContext::for_tests(2).with_memory_limit(Some(4));
        let ((first, nested), captured) = with_context_and_failure(&outer, || {
            let first = acquire(9).unwrap().unwrap_err();
            let ((), nested) = with_context_and_failure(&inner, || {
                assert!(acquire(5).unwrap().is_err());
                assert!(acquire(6).unwrap().is_err());
            });
            assert!(acquire(8).unwrap().is_ok());
            (first, nested)
        });
        assert_eq!(captured, Some(first));
        assert!(matches!(
            nested,
            Some(AdmissionError::ExceedsRemainder { requested_bytes: 5, .. })
        ));
        assert!(acquire(1).is_none(), "no completed scope remains installed");
    }

    #[test]
    fn numerical_failure_capture_follows_worker_tokens_and_isolates_concurrent_runs() {
        std::thread::scope(|threads| {
            let callers: Vec<_> = (1..=2)
                .map(|limit| {
                    threads.spawn(move || {
                        let ctx = ExecutionContext::for_tests(limit).with_memory_limit(Some(limit));
                        let (error, captured) = with_context_and_failure(&ctx, || {
                            let token = NumericalAdmissionToken::capture();
                            std::thread::scope(|workers| {
                                let worker = workers.spawn(|| {
                                    token.execute(&ctx, || acquire(limit + 1).unwrap().unwrap_err())
                                });
                                worker.join().unwrap()
                            })
                        });
                        assert_eq!(captured, Some(error));
                        assert!(matches!(error, AdmissionError::ExceedsRemainder {
                        requested_bytes, ..
                    } if requested_bytes == limit + 1));
                        assert!(acquire(1).is_none());
                    })
                })
                .collect();
            for caller in callers {
                caller.join().unwrap();
            }
        });
    }

    #[test]
    fn numerical_failure_scope_restores_after_unwinding() {
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_context_and_failure(&ctx, || {
                assert!(acquire(1).unwrap().is_err());
                panic!("interrupted component");
            });
        }));
        assert!(result.is_err());
        assert!(acquire(1).is_none());
    }
}
