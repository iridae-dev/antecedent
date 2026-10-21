//! Non-authoritative accounting of operations begun by executing source engines.
//! Successful scientific receipts remain separate from failed or cancelled attempts.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::{Cell, RefCell};
use std::sync::{
    Arc, Mutex,
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
    physical: [u64; 10],
    physical_seen: u64,
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

/// Instrumented physical events. Absence remains unknown rather than a measured zero.
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum PhysicalEvent {
    /// One pass over source rows or validity buffers.
    Scan = 0,
    /// One copied numeric buffer.
    Copy = 1,
    /// Bytes copied between owned numeric buffers.
    CopiedBytes = 2,
    /// A retained numerical cache entry was served.
    CacheHit = 3,
    /// Bytes retained in an executing product.
    RetainedBytes = 4,
    /// Bytes actually written to spill storage.
    SpillBytes = 5,
    /// One completed conditional-independence test.
    ConditionalIndependenceTest = 8,
    /// Completed enclosing model fit, excluding internal numerical solves.
    ModelFit = 9,
}

fn checked_increment(counter: &AtomicU64, amount: u64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| old.checked_add(amount))
        .expect("execution measurement overflow")
        .checked_add(amount)
        .expect("checked sum")
}

/// Record an event at the executing source boundary, never from a planner estimate.
#[doc(hidden)]
pub fn record_physical(event: PhysicalEvent, amount: u64) {
    OBSERVERS.with(|observers| {
        for counters in observers.borrow().iter() {
            checked_increment(&counters.physical[event as usize], amount);
            counters.physical_seen.fetch_or(1 << event as usize, Ordering::Relaxed);
        }
    });
}

/// Lifetime of a root admission reservation observed on this execution.
#[doc(hidden)]
#[derive(Debug)]
pub struct ObservedAdmission {
    counters: Vec<Arc<Counters>>,
    threads: u64,
    bytes: u64,
}
impl ObservedAdmission {
    pub(crate) fn split(&mut self, threads: u32, bytes: u64) -> Self {
        self.threads =
            self.threads.checked_sub(u64::from(threads)).expect("admission split threads");
        self.bytes = self.bytes.checked_sub(bytes).expect("admission split bytes");
        Self { counters: self.counters.clone(), threads: u64::from(threads), bytes }
    }
    pub(crate) fn merge(&mut self, child: &mut Self) {
        self.threads = self.threads.checked_add(child.threads).expect("admission merge threads");
        self.bytes = self.bytes.checked_add(child.bytes).expect("admission merge bytes");
        child.threads = 0;
        child.bytes = 0;
    }
    pub(crate) fn new(threads: u32, bytes: u64) -> Self {
        let counters = OBSERVERS.with(|observers| observers.borrow().clone());
        for counter in &counters {
            let mut held =
                counter.admitted.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            held.0 += u128::from(bytes);
            held.1 += u128::from(threads);
            for (slot, value) in [(6, held.0), (7, held.1)] {
                if let Ok(value) = u64::try_from(value) {
                    counter.physical[slot].fetch_max(value, Ordering::Relaxed);
                    counter.physical_seen.fetch_or(1 << slot, Ordering::Relaxed);
                } else {
                    counter.physical_unavailable.fetch_or(1 << slot, Ordering::Relaxed);
                }
            }
        }
        Self { counters, threads: u64::from(threads), bytes }
    }
}
impl Drop for ObservedAdmission {
    fn drop(&mut self) {
        for counter in &self.counters {
            let mut held =
                counter.admitted.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            held.0 = held
                .0
                .checked_sub(u128::from(self.bytes))
                .expect("admission observation byte balance");
            held.1 = held
                .1
                .checked_sub(u128::from(self.threads))
                .expect("admission observation worker balance");
        }
    }
}

impl AttemptReport {
    /// A measurement when its source boundary was instrumented, or explicit absence.
    #[must_use]
    pub fn physical(&self, event: PhysicalEvent) -> Option<u64> {
        (self.physical_seen & (1 << event as usize) != 0).then_some(self.physical[event as usize])
    }
    /// Peak root admission reservations, not process RSS.
    #[must_use]
    pub fn peak_admitted_bytes(&self) -> Option<u64> {
        (self.physical_seen & (1 << 6) != 0).then_some(self.physical[6])
    }
    /// Peak CPU slots reserved by the run, not an OS thread census.
    #[must_use]
    pub fn peak_admitted_threads(&self) -> Option<u64> {
        (self.physical_seen & (1 << 7) != 0).then_some(self.physical[7])
    }
    /// Native execution facts, preserving the distinction from caller declarations.
    #[must_use]
    pub fn execution_facts(
        &self,
        ctx: &crate::ExecutionContext,
        status: crate::execution_facts::ExecutionFactStatus,
    ) -> crate::execution_facts::ExecutionFacts {
        use crate::execution_facts::{
            ExecutionBackend, ExecutionCounts, ExecutionFacts, ExecutionResources,
            MeasuredExecution, OutputRetention,
        };
        let fits = self.physical[9];
        let counts = ExecutionCounts::new(
            self.physical[0],
            self.physical[1],
            fits,
            self.physical[8],
            self.physical[3],
        );
        ExecutionFacts::from_native(
            MeasuredExecution::new(
                counts,
                OutputRetention::SummaryOnly,
                ExecutionResources::new(self.physical[7], self.physical[6], self.physical[5]),
                ExecutionBackend::new(
                    if self.counts(Operation::ProviderInvocation).attempted > 0 {
                        "native_and_external_provider".into()
                    } else {
                        "native_cpu".into()
                    },
                    ctx.kernel_policy,
                ),
                status,
            ),
            self.physical_seen,
        )
    }
}

/// Original result and separately observed work, including on an error.
#[derive(Debug)]
pub struct ExecutionAttempt<T, E> {
    /// Unchanged result from the actual original operation.
    pub result: Result<T, E>,
    /// Actual component work; this does not issue a successful receipt on failure.
    pub report: AttemptReport,
    /// Engine-produced execution facts when a caller context was supplied.
    pub facts: Option<crate::execution_facts::ExecutionFacts>,
}
#[derive(Default, Debug)]
struct Counters {
    physical: [AtomicU64; 10],
    physical_seen: AtomicU64,
    admitted: Mutex<(u128, u128)>,
    physical_unavailable: AtomicU64,
    attempted: [AtomicU64; 18],
    completed: [AtomicU64; 18],
    failed: [AtomicU64; 18],
}
impl Counters {
    fn snapshot(&self) -> AttemptReport {
        AttemptReport {
            physical: std::array::from_fn(|i| self.physical[i].load(Ordering::Relaxed)),
            physical_seen: self.physical_seen.load(Ordering::Relaxed)
                & !self.physical_unavailable.load(Ordering::Relaxed),
            counts: std::array::from_fn(|i| OperationCounts {
                attempted: self.attempted[i].load(Ordering::Relaxed),
                completed: self.completed[i].load(Ordering::Relaxed),
                failed: self.failed[i].load(Ordering::Relaxed),
            }),
        }
    }
}
thread_local! { static MODEL_FIT_DEPTH: Cell<u32> = const { Cell::new(0) }; }
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
    ExecutionAttempt { result, report: counts.snapshot(), facts: None }
}
/// Observe a public execution, carrying its admission policy into original kernels.
/// Facts survive failure and cancellation; they grant no scientific authority.
pub fn observe_with_context<T, E>(
    ctx: &crate::ExecutionContext,
    execute: impl FnOnce() -> Result<T, E>,
) -> ExecutionAttempt<T, E> {
    let mut attempt = observe_execution(|| crate::numerical_admission::with_context(ctx, execute));
    let status = if ctx.cancellation.is_requested() {
        crate::execution_facts::ExecutionFactStatus::Cancelled
    } else if attempt.result.is_ok() {
        crate::execution_facts::ExecutionFactStatus::Completed
    } else {
        crate::execution_facts::ExecutionFactStatus::Failed
    };
    attempt.facts = Some(attempt.report.execution_facts(ctx, status));
    attempt
}

/// Instrument one original fallible source operation, without changing its result.
// doc(hidden): sibling crates wrap their operations in it; not for downstream code.
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
// doc(hidden): sibling crates and the python bindings carry it across worker threads; not for downstream code.
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
// doc(hidden): sibling crates open an operation scope with it; not for downstream code.
#[doc(hidden)]
pub struct OperationGuard {
    operation: Operation,
    counters: Vec<Arc<Counters>>,
    completed: bool,
    model_fit: bool,
    enclosing_model_fit: bool,
}
impl OperationGuard {
    /// Begin an actual original component invocation, after request preflight when appropriate.
    #[must_use]
    #[doc(hidden)]
    pub fn begin(operation: Operation) -> Self {
        let model_fit = matches!(
            operation,
            Operation::AdjustedFit
                | Operation::GlmFit
                | Operation::LearnerFit
                | Operation::PosteriorFit
        );
        let enclosing_model_fit = model_fit && MODEL_FIT_DEPTH.with(|depth| depth.get() == 0);
        if model_fit {
            MODEL_FIT_DEPTH.with(|depth| {
                depth.set(depth.get().checked_add(1).expect("model fit nesting overflow"))
            });
        }
        let counters = OBSERVERS.with(|observers| observers.borrow().clone());
        for counter in &counters {
            if model_fit {
                counter.physical_seen.fetch_or(1 << 9, Ordering::Relaxed);
            }
            counter.attempted[operation as usize].fetch_add(1, Ordering::Relaxed);
        }
        Self { operation, counters, completed: false, model_fit, enclosing_model_fit }
    }
    /// Mark actual successful completion, consuming the invocation guard.
    #[doc(hidden)]
    pub fn complete(mut self) {
        self.completed = true;
        for counter in &self.counters {
            if self.enclosing_model_fit {
                checked_increment(&counter.physical[9], 1);
            }
            counter.completed[self.operation as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl Drop for OperationGuard {
    fn drop(&mut self) {
        if self.model_fit {
            MODEL_FIT_DEPTH.with(|depth| {
                depth.set(depth.get().checked_sub(1).expect("model fit nesting balance"))
            });
        }
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

#[cfg(test)]
mod physical_observation_tests {
    use super::*;
    use crate::ExecutionContext;
    use crate::execution::AdmissionPool;

    #[test]
    fn admission_tracks_root_reservations_and_child_release_without_double_counting() {
        let pool = AdmissionPool::fixed(4, 100);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool);
        let observed = observe_execution(|| -> Result<(), crate::execution::AdmissionError> {
            let parent = ctx.acquire_admission(2, 80)?;
            let child = parent.acquire_child(1, 30)?;
            drop(parent);
            let other = ctx.acquire_admission(2, 60)?;
            drop(child);
            drop(other);
            Ok(())
        });
        assert!(observed.result.is_ok());
        assert_eq!(observed.report.peak_admitted_bytes(), Some(90));
        assert_eq!(observed.report.peak_admitted_threads(), Some(3));
        assert_eq!(observed.report.physical(PhysicalEvent::Scan), None);
        assert_eq!(ctx.admission().available().bytes(), 100);
    }

    #[test]
    fn worker_events_are_observed_once_and_failure_keeps_native_facts() {
        let ctx = ExecutionContext::production(1, 2);
        let observed = observe_execution(|| -> Result<(), crate::execution::AdmissionError> {
            ctx.map_indexed(4, |_, _| {
                record_physical(PhysicalEvent::Scan, 1);
                Ok::<_, crate::execution::AdmissionError>(())
            })?;
            Err(crate::execution::AdmissionError::Cancelled)
        });
        assert!(observed.result.is_err());
        assert_eq!(observed.report.physical(PhysicalEvent::Scan), Some(4));
        let facts = observed
            .report
            .execution_facts(&ctx, crate::execution_facts::ExecutionFactStatus::Cancelled);
        assert!(facts.is_native());
    }
}
