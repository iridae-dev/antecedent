//! Scoped instrumentation of factories resolved through the public learner specification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use antecedent_core::ExecutionContext;

use crate::{
    DesignView, FittedPredictor, LearnError, LearnerCapabilities, LearnerFactory, PredictionTask,
    TargetView,
};

thread_local! {
    static ACTIVE: RefCell<Option<Arc<AtomicU64>>> = const { RefCell::new(None) };
    static OBSERVERS: RefCell<Vec<Arc<AtomicU64>>> = const { RefCell::new(Vec::new()) };
}

struct Restore(Option<Arc<AtomicU64>>);

impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// Execute an operation and count successful fits of factories resolved inside it.
///
/// Factories carry the counter across worker threads. Already-resolved factories,
/// custom factories and direct concrete-learner calls are outside this instrument.
/// Nested scopes are independent; unwinding restores the enclosing scope.
/// Join worker operations before returning: the result counts fits completed by return.
pub fn count_resolved_fits<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    let count = Arc::new(AtomicU64::new(0));
    let _restore = Restore(ACTIVE.with(|slot| slot.replace(Some(Arc::clone(&count)))));
    let result = operation();
    (result, count.load(Ordering::Relaxed))
}

/// Observe successful resolved factory fits, including nested independent measurement scopes.
/// Already resolved and direct concrete factories remain outside this instrument. Factories
/// carry observers across worker threads; join workers before returning.
pub fn observe_resolved_fits<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    struct RestoreObservers(usize);
    impl Drop for RestoreObservers {
        fn drop(&mut self) {
            OBSERVERS.with(|observers| observers.borrow_mut().truncate(self.0));
        }
    }
    let count = Arc::new(AtomicU64::new(0));
    let _restore = RestoreObservers(OBSERVERS.with(|observers| {
        let mut observers = observers.borrow_mut();
        let before = observers.len();
        observers.push(Arc::clone(&count));
        before
    }));
    let result = operation();
    (result, count.load(Ordering::Relaxed))
}
pub(crate) fn instrument(factory: Box<dyn LearnerFactory>) -> Box<dyn LearnerFactory> {
    let mut counts = OBSERVERS.with(|observers| observers.borrow().clone());
    if let Some(count) = ACTIVE.with(|slot| slot.borrow().clone()) {
        counts.push(count);
    }
    let attempt_observer = antecedent_core::execution_attempt::ObservationToken::capture();
    if counts.is_empty() && attempt_observer.is_empty() {
        factory
    } else {
        Box::new(CountedFactory { factory, counts, attempt_observer })
    }
}
struct CountedFactory {
    factory: Box<dyn LearnerFactory>,
    counts: Vec<Arc<AtomicU64>>,
    attempt_observer: antecedent_core::execution_attempt::ObservationToken,
}

impl LearnerFactory for CountedFactory {
    fn task(&self) -> PredictionTask {
        self.factory.task()
    }
    fn capabilities(&self) -> LearnerCapabilities {
        self.factory.capabilities()
    }
    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        self.attempt_observer.execute(|| {
            antecedent_core::execution_attempt::run_operation(
                antecedent_core::execution_attempt::Operation::LearnerFit,
                || {
                    let result = self.factory.fit(x, y, weights, ctx)?;
                    for count in &self.counts {
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(result)
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LearnerSpec, LinearSpec, resolve};

    #[test]
    fn attempt_observer_alone_tracks_real_resolved_worker_success_and_failure() {
        use antecedent_core::execution_attempt::{Operation, observe_execution};
        let attempted = observe_execution(|| {
            // No legacy fit-count scope is installed: the attempt token alone
            // must force the original resolved factory to carry observation.
            let factory = resolve(LearnerSpec::Linear(LinearSpec::default())).unwrap();
            std::thread::spawn(move || {
                let ctx = ExecutionContext::for_tests(11);
                let y = [1.0, 3.0, 5.0];
                factory
                    .fit(
                        DesignView::from_column_major(&[1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2)
                            .unwrap(),
                        TargetView::new(&y),
                        None,
                        &ctx,
                    )
                    .unwrap();
                assert!(
                    factory
                        .fit(
                            DesignView::from_column_major(&[1.0; 6], 3, 2).unwrap(),
                            TargetView::new(&y),
                            None,
                            &ctx,
                        )
                        .is_err()
                );
            })
            .join()
            .unwrap();
            Ok::<_, ()>(())
        });
        let fits = attempted.report.counts(Operation::LearnerFit);
        assert_eq!((fits.attempted, fits.completed, fits.failed), (2, 1, 1));
        let solves = attempted.report.counts(Operation::LeastSquaresSolve);
        assert_eq!((solves.attempted, solves.completed, solves.failed), (2, 1, 1));
        assert!(attempted.report.is_complete());
    }

    fn fit() {
        let x = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 5.0];
        resolve(LearnerSpec::Linear(LinearSpec::default()))
            .unwrap()
            .fit(
                DesignView::from_column_major(&x, 3, 2).unwrap(),
                TargetView::new(&y),
                None,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
    }

    #[test]
    fn nested_and_unwinding_scopes_restore_the_parent_counter() {
        let ((), outer) = count_resolved_fits(|| {
            fit();
            let ((), inner) = count_resolved_fits(fit);
            assert_eq!(inner, 1);
            let failed = std::panic::catch_unwind(|| count_resolved_fits(|| panic!("test")));
            assert!(failed.is_err());
            fit();
        });
        assert_eq!(outer, 2);
        let ((), clean) = count_resolved_fits(|| {});
        assert_eq!(clean, 0);
    }

    #[test]
    fn cumulative_observers_include_nested_independent_scopes_and_restore_after_unwind() {
        let ((), observed) = observe_resolved_fits(|| {
            let ((), outer) = count_resolved_fits(|| {
                fit();
                let ((), inner) = count_resolved_fits(fit);
                assert_eq!(inner, 1);
                let ((), nested) = observe_resolved_fits(fit);
                assert_eq!(nested, 1);
                let failed = std::panic::catch_unwind(|| observe_resolved_fits(|| panic!("test")));
                assert!(failed.is_err());
                fit();
            });
            assert_eq!(outer, 3);
        });
        assert_eq!(observed, 4);
        let ((), clean) = observe_resolved_fits(|| {});
        assert_eq!(clean, 0);
    }

    #[test]
    fn resolved_factory_carries_successful_fit_counter_to_a_worker() {
        let ((), fits) = count_resolved_fits(|| {
            let factory = resolve(LearnerSpec::Linear(LinearSpec::default())).unwrap();
            std::thread::spawn(move || {
                let x = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
                factory
                    .fit(
                        DesignView::from_column_major(&x, 3, 2).unwrap(),
                        TargetView::new(&[1.0, 3.0, 5.0]),
                        None,
                        &ExecutionContext::for_tests(1),
                    )
                    .unwrap();
            })
            .join()
            .unwrap();
        });
        assert_eq!(fits, 1);
    }

    #[test]
    fn failed_fit_does_not_count_as_a_completed_model() {
        let ((), fits) = count_resolved_fits(|| {
            let factory = resolve(LearnerSpec::Linear(LinearSpec::default())).unwrap();
            let design = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
            assert!(
                factory
                    .fit(
                        DesignView::from_column_major(&design, 3, 2).unwrap(),
                        TargetView::new(&[1.0]),
                        None,
                        &ExecutionContext::for_tests(1)
                    )
                    .is_err()
            );
        });
        assert_eq!(fits, 0);
    }
}
