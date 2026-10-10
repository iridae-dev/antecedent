//! Non-authoritative accounting of operations begun by executing source engines.
//! Successful scientific receipts remain separate from failed or cancelled attempts.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Distinct executing components; a numerical solve and a enclosing model fit are not additive work units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum Operation {
    /// Original causal identification invocation, including a refused proof.
    Identification,
    /// Original dense numerical solve.
    LeastSquaresSolve,
    /// Original GLM numerical fit.
    GlmFit,
    /// Original adjusted fitted-model construction.
    AdjustedFit,
    /// Resolved nuisance/final learner factory invocation.
    LearnerFit,
    /// Original conjugate posterior parameter solve.
    PosteriorFit,
    /// One original complete aligned posterior draw emission.
    PosteriorDraw,
    /// Original score table construction.
    ScoreConstruction,
    /// Original empirical factor/history mechanism construction.
    FactorConstruction,
    /// Original expression compiler invocation.
    ProgramCompilation,
    /// Original numerical provider construction/binding.
    ProviderBinding,
    /// Original uncached numerical factor evaluation.
    FactorEvaluation,
    /// Original complete functional integration or mean projection.
    Integration,
    /// Original numerical or foreign provider invocation.
    ProviderInvocation,
    /// Original retained law summary operation.
    LawSummary,
    /// Original terminal decision invocation.
    Decision,
    /// Original cache served a previously retained failure; no fit was attempted.
    CachedFailureRead,
    /// Original prior parameter construction or declared prior composition.
    PriorConstruction,
}
const OPERATIONS: [Operation; 18] = [
    Operation::Identification,
    Operation::LeastSquaresSolve,
    Operation::GlmFit,
    Operation::AdjustedFit,
    Operation::LearnerFit,
    Operation::PosteriorFit,
    Operation::PosteriorDraw,
    Operation::ScoreConstruction,
    Operation::FactorConstruction,
    Operation::ProgramCompilation,
    Operation::ProviderBinding,
    Operation::FactorEvaluation,
    Operation::Integration,
    Operation::ProviderInvocation,
    Operation::LawSummary,
    Operation::Decision,
    Operation::CachedFailureRead,
    Operation::PriorConstruction,
];
/// Counts for one operation kind. A failed invocation is never a successful fit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperationCounts {
    /// Actual component invocations begun.
    pub attempted: u64,
    /// Actual component invocations completed successfully.
    pub completed: u64,
    /// Begun component invocations that returned or unwound without success.
    pub failed: u64,
}
impl OperationCounts {
    /// Work still running at observation time; callers must join workers before returning.
    #[must_use]
    pub fn unfinished(self) -> u64 {
        self.attempted.saturating_sub(self.completed.saturating_add(self.failed))
    }
}
/// Read-only execution observation. It is not a scientific claim, resumable state, or calibration evidence.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AttemptReport {
    counts: [OperationCounts; 18],
}
impl AttemptReport {
    /// Counts of a specific original component operation.
    #[must_use]
    pub fn counts(&self, operation: Operation) -> OperationCounts {
        self.counts[operation as usize]
    }
    /// All operation kinds in stable order, including zero counts.
    pub fn operations(&self) -> impl Iterator<Item = (Operation, OperationCounts)> + '_ {
        OPERATIONS.into_iter().map(|operation| (operation, self.counts(operation)))
    }
    /// Whether this execution entered no instrumented original component.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|c| c.attempted == 0)
    }
    /// Whether all begun operations finished before observation returned.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.counts.iter().all(|c| c.unfinished() == 0)
    }
}
/// Original result and separately observed work, including on an error.
#[derive(Debug)]
pub struct ExecutionAttempt<T, E> {
    /// Unchanged result from the actual original operation.
    pub result: Result<T, E>,
    /// Actual component work; this does not issue a successful receipt on failure.
    pub report: AttemptReport,
}
#[derive(Default)]
struct Counters {
    attempted: [AtomicU64; 18],
    completed: [AtomicU64; 18],
    failed: [AtomicU64; 18],
}
impl Counters {
    fn snapshot(&self) -> AttemptReport {
        AttemptReport {
            counts: std::array::from_fn(|i| OperationCounts {
                attempted: self.attempted[i].load(Ordering::Relaxed),
                completed: self.completed[i].load(Ordering::Relaxed),
                failed: self.failed[i].load(Ordering::Relaxed),
            }),
        }
    }
}
thread_local! { static OBSERVERS: RefCell<Vec<Arc<Counters>>> = const { RefCell::new(Vec::new()) }; }
struct Scope {
    previous: usize,
}
impl Drop for Scope {
    fn drop(&mut self) {
        OBSERVERS.with(|observers| observers.borrow_mut().truncate(self.previous));
    }
}
/// Observe actual source work, preserving the original Result on success or failure.
/// Nested observers each see the same actual operation once. Worker operations must be joined.
/// Preflight refusals and complete reuse produce zero counts when no component is entered.
pub fn observe_execution<T, E>(execute: impl FnOnce() -> Result<T, E>) -> ExecutionAttempt<T, E> {
    let counts = Arc::new(Counters::default());
    let _scope = OBSERVERS.with(|observers| {
        let mut observers = observers.borrow_mut();
        let previous = observers.len();
        observers.push(Arc::clone(&counts));
        Scope { previous }
    });
    let result = execute();
    ExecutionAttempt { result, report: counts.snapshot() }
}
/// Instrument one original fallible source operation, without changing its result.
#[doc(hidden)]
pub fn run_operation<T, E>(
    operation: Operation,
    execute: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let guard = OperationGuard::begin(operation);
    let result = execute();
    if result.is_ok() {
        guard.complete();
    }
    result
}
/// Capture source execution observation for an actual factory dispatched to a worker.
/// This token carries counters only and cannot provide scientific execution authority.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct ObservationToken {
    counters: Vec<Arc<Counters>>,
}
impl ObservationToken {
    /// Capture current nested observers when the executing factory is resolved.
    #[must_use]
    #[doc(hidden)]
    pub fn capture() -> Self {
        Self { counters: OBSERVERS.with(|observers| observers.borrow().clone()) }
    }
    /// Whether capture retained any active operation observer.
    #[doc(hidden)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty()
    }
    /// Carry observation through the actual worker call without duplicating same-thread counts.
    #[doc(hidden)]
    pub fn execute<T>(&self, execute: impl FnOnce() -> T) -> T {
        let _scope = OBSERVERS.with(|observers| {
            let mut observers = observers.borrow_mut();
            let previous = observers.len();
            for counter in &self.counters {
                if !observers.iter().any(|existing| Arc::ptr_eq(existing, counter)) {
                    observers.push(Arc::clone(counter));
                }
            }
            Scope { previous }
        });
        execute()
    }
}
/// Source-engine invocation guard; failure/unwind cannot manufacture a successful operation.
#[doc(hidden)]
pub struct OperationGuard {
    operation: Operation,
    counters: Vec<Arc<Counters>>,
    completed: bool,
}
impl OperationGuard {
    /// Begin an actual original component invocation, after request preflight when appropriate.
    #[must_use]
    #[doc(hidden)]
    pub fn begin(operation: Operation) -> Self {
        let counters = OBSERVERS.with(|observers| observers.borrow().clone());
        for counter in &counters {
            counter.attempted[operation as usize].fetch_add(1, Ordering::Relaxed);
        }
        Self { operation, counters, completed: false }
    }
    /// Mark actual successful completion, consuming the invocation guard.
    #[doc(hidden)]
    pub fn complete(mut self) {
        self.completed = true;
        for counter in &self.counters {
            counter.completed[self.operation as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl Drop for OperationGuard {
    fn drop(&mut self) {
        if !self.completed {
            for counter in &self.counters {
                counter.failed[self.operation as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn result_failure_and_nested_observers_keep_operation_semantics() {
        let outer = observe_execution(|| {
            let inner = observe_execution(|| -> Result<(), &str> {
                let _failure = OperationGuard::begin(Operation::LeastSquaresSolve);
                OperationGuard::begin(Operation::Identification).complete();
                Err("rank deficient")
            });
            assert_eq!(inner.result, Err("rank deficient"));
            assert_eq!(
                inner.report.counts(Operation::LeastSquaresSolve),
                OperationCounts { attempted: 1, completed: 0, failed: 1 }
            );
            Ok::<_, &str>(inner.report)
        });
        assert_eq!(outer.report, outer.result.unwrap());
        assert!(outer.report.is_complete());
        assert!(observe_execution(|| Err::<(), _>("preflight")).report.is_empty());
    }
    #[test]
    fn resolved_worker_and_same_thread_tokens_do_not_duplicate_counts() {
        let observation = observe_execution(|| {
            let token = ObservationToken::capture();
            token.execute(|| OperationGuard::begin(Operation::LearnerFit).complete());
            std::thread::spawn(move || {
                token.execute(|| {
                    let _failed = OperationGuard::begin(Operation::LearnerFit);
                })
            })
            .join()
            .unwrap();
            Ok::<_, ()>(())
        });
        assert_eq!(
            observation.report.counts(Operation::LearnerFit),
            OperationCounts { attempted: 2, completed: 1, failed: 1 }
        );
        assert!(observation.report.is_complete());
    }
    #[test]
    fn unwinding_restores_observers_and_marks_begun_operations_failed() {
        let attempt = observe_execution(|| {
            let panic = std::panic::catch_unwind(|| {
                observe_execution(|| -> Result<(), ()> {
                    let _started = OperationGuard::begin(Operation::PosteriorFit);
                    panic!("engine panic");
                })
            });
            assert!(panic.is_err());
            Ok::<_, ()>(())
        });
        assert_eq!(attempt.report.counts(Operation::PosteriorFit).failed, 1);
        assert!(observe_execution(|| Ok::<_, ()>(())).report.is_empty());
    }
}
