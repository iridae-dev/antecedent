//! Ctrl-C and callback-exception propagation for native runs that have released the GIL.
//!
//! CPython delivers `SIGINT` by setting a pending flag that only the *main thread*, holding the
//! GIL, can turn into `KeyboardInterrupt`. A native run that detached from the GIL therefore never
//! sees it. [`run`] keeps the calling (Python) thread as the watcher: the native work runs on a
//! scoped worker thread while the caller wakes every [`POLL`], re-acquires the GIL, and calls
//! `check_signals`. A pending signal cancels the run's [`CancellationToken`] (the same token the
//! engines already poll cooperatively) and the `KeyboardInterrupt` is what the caller gets back.
//!
//! Python callables invoked from native code fail with a `PyErr` that would otherwise be reduced
//! to a message. [`callback_failure`] keeps the original exception under an id its message
//! carries; [`attribute`] looks it up by that id and re-raises it
//! (`KeyboardInterrupt`, `SystemExit`, ... unchanged) or chains it as `__cause__` of the domain
//! error, so tracebacks into user code survive and `except CausalError` cannot swallow an
//! interrupt. Each native run owns its callback ids and releases any unattributed
//! failures when the run finishes, including successful fallbacks and signal overrides.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use antecedent_core::{CancellationToken, WeakCancellationToken};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::{CausalError, catch_ffi};

/// How often the calling thread re-acquires the GIL to look for a pending signal.
const POLL: Duration = Duration::from_millis(50);

/// Native analyses recurse (graph search, tree builders); the worker gets a stack no smaller than
/// the 8 MiB a Python main thread has by default, with headroom for the engines' own frames.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Retained callback failures. A full map refuses a new id instead of dropping
/// the oldest, which may still be waiting to be attributed.
const MAX_STASHED: usize = 256;

/// Tokens the interrupt loop cancels for one run.
#[derive(Default)]
struct Scope {
    own: CancellationToken,
    linked: Mutex<Vec<CancellationToken>>,
    /// IDs retained by callbacks on this worker, including errors a native fallback
    /// suppresses or a pending signal supersedes before attribution.
    failures: Mutex<Vec<u64>>,
    /// First observer exception or callback BaseException. Native success/fallback
    /// must not swallow an observer failure or a request to interrupt or exit.
    /// Bounded independently of STASH.
    interruption: Mutex<Option<PyErr>>,
}

impl Scope {
    fn cancel_all(&self) {
        self.own.cancel();
        for token in self.linked.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            token.cancel();
        }
    }

    fn release_failures(&self) {
        let ids =
            std::mem::take(&mut *self.failures.lock().unwrap_or_else(PoisonError::into_inner));
        // Never release Python exceptions while holding the stash mutex: their
        // destruction may execute Python finalizers and re-enter a callback.
        let discarded = {
            let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
            ids.into_iter().filter_map(|id| stash.remove(&id)).collect::<Vec<_>>()
        };
        drop(discarded);
        let superseded = self.interruption.lock().unwrap_or_else(PoisonError::into_inner).take();
        drop(superseded);
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        self.release_failures();
    }
}

/// Rebind the run's explicit callback ownership while a native progress worker executes.
struct ActiveGuard(Option<Arc<Scope>>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = self.0.take());
    }
}

struct ScopedProgress {
    inner: Arc<dyn antecedent_core::ProgressSink>,
    scope: std::sync::Weak<Scope>,
}

impl antecedent_core::ProgressSink for ScopedProgress {
    fn report(&self, fraction: f64, stage: &str) {
        // A retained context must not keep the completed run (and its Python
        // failures) alive or call its observer after that run has ended.
        let Some(scope) = self.scope.upgrade() else { return };
        let _restore = ActiveGuard(ACTIVE.with(|active| active.replace(Some(scope))));
        self.inner.report(fraction, stage);
    }
}

/// Carry the executing run's ownership into progress callbacks on native pool threads.
///
/// Progress-bearing Python contexts are constructed on the active worker by
/// ClickControls::ctx or PreparedOptions::ctx, once for each click/prepare. They
/// wrap the raw observer afresh on every run; observer reuse never reuses an old
/// run's binding. Contexts constructed before a run in inspection/export have no
/// progress observer, so cancellation's separate pending-token queue suffices.
pub(crate) fn bind_progress(
    progress: Option<Arc<dyn antecedent_core::ProgressSink>>,
) -> Option<Arc<dyn antecedent_core::ProgressSink>> {
    let inner = progress?;
    let scope = ACTIVE.with(|active| active.borrow().as_ref().map(Arc::downgrade));
    match scope {
        Some(scope) => Some(Arc::new(ScopedProgress { inner, scope })),
        None => Some(inner),
    }
}

/// Queue length at which dead entries are first pruned from [`PENDING`].
const PENDING_PRUNE_FLOOR: usize = 64;

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Scope>>> = const { RefCell::new(None) };
    /// Contexts built on a Python thread ahead of the [`run`] that will execute them.
    ///
    /// Weak: a context built and dropped without running leaves nothing to cancel, and a
    /// live one is never evicted or refused however many wait.
    static PENDING: RefCell<Vec<WeakCancellationToken>> = const { RefCell::new(Vec::new()) };
    /// Queue length at which [`PENDING`] is next pruned (doubles with the live set).
    static PENDING_PRUNE_AT: Cell<usize> = const { Cell::new(PENDING_PRUNE_FLOOR) };
}

/// The interruptible run's token, if the current thread is a [`run`] worker.
pub(crate) fn ambient_token() -> Option<CancellationToken> {
    ACTIVE.with(|active| active.borrow().as_ref().map(|scope| scope.own.clone()))
}

/// Cancel `token` too when the current run is interrupted (an explicit user token stays usable).
///
/// Called on a worker, the token joins that run; called on a Python thread that has not yet
/// entered [`run`] (a context built ahead of the detached call), it joins the next run there.
pub(crate) fn link(token: &CancellationToken) {
    let joined = ACTIVE.with(|active| {
        active.borrow().as_ref().is_some_and(|scope| {
            scope.linked.lock().unwrap_or_else(PoisonError::into_inner).push(token.clone());
            true
        })
    });
    if !joined {
        PENDING.with(|pending| {
            let mut pending = pending.borrow_mut();
            if pending.len() >= PENDING_PRUNE_AT.get() {
                pending.retain(WeakCancellationToken::is_alive);
                PENDING_PRUNE_AT.set((pending.len() * 2).max(PENDING_PRUNE_FLOOR));
            }
            pending.push(token.downgrade());
        });
    }
}

/// Callback failures awaiting attribution, by the id their message carries.
static STASH: Mutex<BTreeMap<u64, PyErr>> = Mutex::new(BTreeMap::new());

/// Source of callback-failure ids; one id per recorded failure, never reused.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

const TAG_OPEN: &str = "[callback failure #";

/// Whether another callback failure can be retained without dropping one already stored.
fn stash_accepts(len: usize) -> bool {
    len < MAX_STASHED
}

/// Record the Python exception behind a callback failure and return its domain error message.
/// Ordinary exceptions receive an id tag for [`attribute`], so concurrent runs cannot
/// misattribute a traceback. A scoped BaseException cancels the run and is retained
/// directly, even if native code converts its error into a report.
pub(crate) fn callback_failure(what: &str, err: PyErr) -> String {
    let scope = ACTIVE.with(|active| active.borrow().clone());
    // No run owns an unscoped failure, so there is no later attribution or cleanup
    // boundary at which retaining its Python exception would be safe.
    let Some(scope) = scope else {
        return unretained_callback_description(&format!("{what}: {err}"));
    };
    {
        let interrupts = Python::attach(|py| !err.is_instance_of::<PyException>(py));
        if interrupts {
            let discarded = {
                let mut first = scope.interruption.lock().unwrap_or_else(PoisonError::into_inner);
                if first.is_none() {
                    *first = Some(err);
                    None
                } else {
                    Some(err)
                }
            };
            drop(discarded);
            scope.cancel_all();
            return format!("{what}: Python callback interrupted the native run");
        }
    }
    // Exception __str__ can execute user Python, including another failing
    // callback. Format ordinary errors before locking, even when the stash is full.
    let description = format!("{what}: {err}");
    let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
    if !stash_accepts(stash.len()) {
        // The oldest entry may still be waiting for [`attribute`]. Dropping it
        // would attach this failure's exception to an earlier run. The new
        // failure stays in the message and is not given an id.
        drop(stash);
        return unretained_callback_description(&description);
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    stash.insert(id, err);
    drop(stash);
    scope.failures.lock().unwrap_or_else(PoisonError::into_inner).push(id);
    format!("{description} {TAG_OPEN}{id}]")
}

/// Retain an observer's original exception at the executing run boundary.
/// Progress/stage traits return `()`, so cancel the native work and let [`run`]
/// raise this exact object, including its Python traceback, even after native success.
/// An observer invoked outside a run has no safe exception owner; return it to
/// the caller rather than retaining it in an unrelated future run.
pub(crate) fn observer_failure(err: PyErr) -> Result<(), PyErr> {
    // Keep the exact exception object and traceback, while distinguishing a user
    // observer's exception from an identically typed native scientific refusal.
    Python::attach(|py| {
        let _ = err.value(py).setattr("_antecedent_operational_callback", true);
    });
    let scope = ACTIVE.with(|active| active.borrow().clone());
    let Some(scope) = scope else { return Err(err) };
    let discarded = {
        let mut first = scope.interruption.lock().unwrap_or_else(PoisonError::into_inner);
        if first.is_none() {
            *first = Some(err);
            None
        } else {
            Some(err)
        }
    };
    drop(discarded);
    scope.cancel_all();
    Ok(())
}

/// An outer error that could not retain its original must not attribute a nested
/// callback's tag copied into its __str__ output. Keep its readable diagnostics,
/// but neutralize the private lookup syntax before returning the unretained error.
fn unretained_callback_description(description: &str) -> String {
    description.replace(TAG_OPEN, "[unretained callback failure #")
}

/// The callback-failure ids a domain error message carries, in order of appearance.
fn tagged_ids(text: &str) -> impl Iterator<Item = u64> + '_ {
    text.match_indices(TAG_OPEN).filter_map(|(at, _)| {
        let rest = &text[at + TAG_OPEN.len()..];
        rest.split_once(']')?.0.parse().ok()
    })
}

/// Replace a domain error that wraps a callback failure with the Python exception behind it.
///
/// A non-`Exception` (`KeyboardInterrupt`, `SystemExit`, ...) is re-raised as is. An `Exception`
/// becomes the `__cause__` of the domain error, keeping its type and traceback.
/// Only this run's registered ids are eligible; nested error descriptions cannot
/// take precedence over the outer callback's final tag.
fn attribute(py: Python<'_>, err: PyErr, scope: &Scope) -> PyErr {
    if !err.is_instance_of::<PyException>(py) {
        return err;
    }
    // Formatting can call Python and register more failures; do it before either
    // lock. The final owned tag belongs to the outer callback, appended after its
    // exception's __str__ returned any nested callback's tagged description.
    let text = err.value(py).to_string();
    let owned = scope.failures.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let original = {
        let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
        let id = tagged_ids(&text).filter(|id| owned.contains(id) && stash.contains_key(id)).last();
        id.and_then(|id| stash.remove(&id))
    };
    let Some(original) = original else { return err };
    if original.is_instance_of::<PyException>(py) {
        err.set_cause(py, Some(original));
        err
    } else {
        original
    }
}

/// Run `f` on a worker thread with the GIL released, so Ctrl-C in the calling thread cancels it.
///
/// Panics become [`CausalError`]. A signal observed while `f` ran always wins over `f`'s own
/// result: the user asked for the run to stop. Without an observed signal, a callback's first
/// non-Exception or observer failure also wins over native fallback/report conversion.
pub(crate) fn run<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    let scope = Arc::new(Scope::default());
    scope.linked.lock().unwrap_or_else(PoisonError::into_inner).extend(PENDING.with(|pending| {
        std::mem::take(&mut *pending.borrow_mut())
            .iter()
            .filter_map(WeakCancellationToken::upgrade)
            .collect::<Vec<_>>()
    }));
    let slot: (Mutex<Option<PyResult<T>>>, Condvar) = (Mutex::new(None), Condvar::new());
    let observation = antecedent_core::execution_attempt::ObservationToken::capture();
    let execution_policy = crate::execution_api::capture();
    // One worker only so the calling thread can poll signals; it runs the same work serially, so no extra parallelism is added and the work admits itself.
    let outcome = std::thread::scope(|s| {
        let worker_scope = Arc::clone(&scope);
        let slot = &slot;
        let spawned =
            std::thread::Builder::new().stack_size(WORKER_STACK_BYTES).spawn_scoped(s, move || {
                let _execution_scope = crate::execution_api::enter(execution_policy);
                ACTIVE.with(|active| *active.borrow_mut() = Some(worker_scope));
                let result = observation.execute(|| catch_ffi(f));
                *slot.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
                slot.1.notify_one();
            });
        if let Err(e) = spawned {
            return Err(CausalError::new_err(format!("could not start the native worker: {e}")));
        }
        let mut interrupt: Option<PyErr> = None;
        let result = loop {
            let finished = py.detach(|| {
                let guard = slot.0.lock().unwrap_or_else(PoisonError::into_inner);
                let (mut guard, _) = slot
                    .1
                    .wait_timeout_while(guard, POLL, |r| r.is_none())
                    .unwrap_or_else(PoisonError::into_inner);
                guard.take()
            });
            if let Some(result) = finished {
                break result;
            }
            if interrupt.is_none() {
                if let Err(e) = py.check_signals() {
                    scope.cancel_all();
                    interrupt = Some(e);
                }
            }
        };
        if let Some(error) = interrupt {
            Err(error)
        } else {
            let original = scope.interruption.lock().unwrap_or_else(PoisonError::into_inner).take();
            match original {
                Some(error) => Err(error),
                None => result,
            }
        }
    });
    let outcome = outcome.map_err(|e| attribute(py, e, &scope));
    // A worker's thread-local Arc may outlive joining its scoped work. Release
    // unconsumed Python failures at this run boundary, after cause attribution,
    // rather than depending on the final Arc/thread-local destructor timing.
    scope.release_failures();
    outcome
}

#[cfg(test)]
mod tests {
    use super::{TAG_OPEN, tagged_ids};

    #[test]
    fn a_domain_error_names_its_callback_failures_by_id_not_by_text() {
        // Two runs whose callbacks fail with byte-identical text carry different ids, so
        // attribution keyed on the id cannot hand one run the other's exception.
        let first = format!("Python CI callback failed: ValueError: boom {TAG_OPEN}7]");
        let second = format!("Python CI callback failed: ValueError: boom {TAG_OPEN}8]");
        assert_ne!(first, second);
        assert_eq!(tagged_ids(&first).collect::<Vec<_>>(), vec![7]);
        assert_eq!(tagged_ids(&second).collect::<Vec<_>>(), vec![8]);
        let wrapped = format!("discovery failed: backend: {second}");
        assert_eq!(tagged_ids(&wrapped).collect::<Vec<_>>(), vec![8]);
        assert_eq!(tagged_ids("no tag here").count(), 0);
        assert_eq!(tagged_ids(&format!("{TAG_OPEN}not a number]")).count(), 0);
    }

    #[test]
    fn a_full_callback_stash_does_not_drop_the_oldest_id() {
        use super::{MAX_STASHED, stash_accepts};
        assert!(stash_accepts(0));
        assert!(stash_accepts(MAX_STASHED - 1));
        assert!(!stash_accepts(MAX_STASHED));
    }

    #[test]
    fn successful_fallbacks_release_callback_exceptions_after_every_run() {
        use super::{MAX_STASHED, STASH, callback_failure, run};
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for _ in 0..=MAX_STASHED {
                let retained = std::sync::Arc::new(std::sync::Mutex::new(None));
                let worker_retained = std::sync::Arc::clone(&retained);
                let id = run(py, || {
                    // Cleanup belongs to the completed run, even when a native
                    // owner still holds its scope after the worker returns.
                    *worker_retained.lock().unwrap() =
                        super::ACTIVE.with(|active| active.borrow().clone());
                    let message =
                        callback_failure("suppressed callback", PyValueError::new_err("boom"));
                    Ok(tagged_ids(&message)
                        .next()
                        .expect("previous runs must release their failures"))
                })
                .unwrap();
                assert!(!STASH.lock().unwrap().contains_key(&id));
                assert!(retained.lock().unwrap().is_some());
            }
        });
    }

    #[test]
    fn callback_base_exceptions_survive_and_superseded_failures_are_released() {
        use super::{STASH, callback_failure, run};
        use crate::CausalError;
        use pyo3::exceptions::{PyKeyboardInterrupt, PySystemExit, PyValueError};
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for original in [PyKeyboardInterrupt::new_err("stop"), PySystemExit::new_err(17)] {
                let expected = original.clone_ref(py);
                let error = run::<_, ()>(py, move || {
                    Err(CausalError::new_err(callback_failure("callback", original)))
                })
                .unwrap_err();
                assert!(error.value(py).is(expected.value(py)));
            }
            let id = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let recorded = std::sync::Arc::clone(&id);
            let error = run::<_, ()>(py, move || {
                let message = callback_failure("discarded callback", PyValueError::new_err("boom"));
                recorded.store(tagged_ids(&message).next().unwrap(), super::Ordering::Relaxed);
                // Mirrors run's signal-wins branch: this result carries no callback tag.
                Err(PyKeyboardInterrupt::new_err("signal wins"))
            })
            .unwrap_err();
            assert!(error.is_instance_of::<PyKeyboardInterrupt>(py));
            assert!(!STASH.lock().unwrap().contains_key(&id.load(super::Ordering::Relaxed)));
        });
    }

    #[test]
    fn observer_exceptions_survive_native_success_with_original_identity() {
        use super::{ambient_token, observer_failure, run};
        use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for original in
                [PyValueError::new_err("observer failed"), PyKeyboardInterrupt::new_err("stop")]
            {
                let expected = original.clone_ref(py);
                let error = run::<_, ()>(py, move || {
                    observer_failure(original)?;
                    assert!(ambient_token().unwrap().is_requested());
                    Ok(())
                })
                .unwrap_err();
                assert!(error.value(py).is(expected.value(py)));
            }
        });
    }

    #[test]
    fn native_success_cannot_suppress_a_callback_interrupt() {
        use super::{ambient_token, callback_failure, link, run};
        use pyo3::exceptions::{PyKeyboardInterrupt, PySystemExit};
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for original in [PyKeyboardInterrupt::new_err("stop"), PySystemExit::new_err(17)] {
                let expected = original.clone_ref(py);
                let error = run::<_, ()>(py, move || {
                    let token = ambient_token().unwrap();
                    let linked = antecedent_core::CancellationToken::new();
                    link(&linked);
                    let _suppressed = callback_failure("ignored native fallback", original);
                    assert!(token.is_cancelled());
                    assert!(linked.is_cancelled());
                    let _later = callback_failure("later failure", PySystemExit::new_err("later"));
                    Ok(())
                })
                .unwrap_err();
                assert!(error.value(py).is(expected.value(py)));
            }
        });
    }

    #[test]
    fn concurrent_identical_callback_text_keeps_each_runs_original_cause() {
        use super::{callback_failure, run};
        use crate::CausalError;
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|threads| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    let barrier = std::sync::Arc::clone(&barrier);
                    threads.spawn(move || {
                        Python::attach(|py| {
                            let expected = PyValueError::new_err("identical callback text");
                            let original = expected.clone_ref(py);
                            let error = run::<_, ()>(py, move || {
                                let message = callback_failure("callback", original);
                                barrier.wait();
                                Err(CausalError::new_err(message))
                            })
                            .unwrap_err();
                            let cause =
                                error.cause(py).expect("ordinary exceptions remain chained");
                            assert!(cause.value(py).is(expected.value(py)));
                        })
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }

    #[test]
    fn native_progress_workers_preserve_each_runs_interrupt_and_restore_their_scope() {
        use super::{ambient_token, run};
        use pyo3::exceptions::PySystemExit;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        Python::initialize();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|threads| {
            let callers: Vec<_> = (0..2)
                .map(|_| {
                    let barrier = std::sync::Arc::clone(&barrier);
                    threads.spawn(move || {
                        Python::attach(|py| {
                            // from_code uses PyImport_ExecCodeModuleEx: repeating a
                            // module name reuses sys.modules and overwrites the other
                            // caller's globals. Give each callback its own dictionary.
                            let globals = PyDict::new(py);
                            let expected = PySystemExit::new_err("identical progress interruption");
                            globals.set_item("original", expected.value(py)).unwrap();
                            py.run(
                                pyo3::ffi::c_str!("def observe(*args):\n    raise original\n"),
                                Some(&globals),
                                None,
                            )
                            .unwrap();
                            let observer: std::sync::Arc<dyn antecedent_core::ProgressSink> =
                                std::sync::Arc::new(crate::callbacks::PyProgressSink::new(
                                    globals.get_item("observe").unwrap().unwrap().unbind(),
                                ));
                            // Reuse the same raw Python observer in two executions; each
                            // context must bind its new scope rather than retain the old run.
                            for execution in 0..2 {
                                let observer = std::sync::Arc::clone(&observer);
                                let barrier = std::sync::Arc::clone(&barrier);
                                let error = run::<_, ()>(py, move || {
                                    let ctx = crate::py_execution_context_ext(
                                        1,
                                        2,
                                        None,
                                        Some(observer),
                                        None,
                                    );
                                    let owner = std::thread::current().id();
                                    if execution == 0 {
                                        barrier.wait();
                                    }
                                    ctx.map_indexed(2, |_, inner| {
                                        assert_ne!(
                                            owner,
                                            std::thread::current().id(),
                                            "must exercise a real native pool worker"
                                        );
                                        assert!(ambient_token().is_none());
                                        inner
                                            .progress
                                            .as_ref()
                                            .unwrap()
                                            .report(0.5, "parallel progress");
                                        assert!(
                                            ambient_token().is_none(),
                                            "adapter restores the prior worker scope"
                                        );
                                        Ok::<(), ()>(())
                                    })
                                    .unwrap();
                                    Ok(())
                                })
                                .unwrap_err();
                                assert!(error.value(py).is(expected.value(py)));
                            }
                            // No sys.modules root or globals/function/traceback cycle
                            // should retain this test's original exception afterward.
                            globals.clear();
                        })
                    })
                })
                .collect();
            for caller in callers {
                caller.join().unwrap();
            }
        });
    }

    #[test]
    fn retained_progress_adapter_does_not_retain_a_completed_run() {
        use super::{ACTIVE, bind_progress, run};
        use pyo3::prelude::*;

        struct Count(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl antecedent_core::ProgressSink for Count {
            fn report(&self, _: f64, _: &str) {
                self.0.fetch_add(1, super::Ordering::Relaxed);
            }
        }
        Python::initialize();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let inner: std::sync::Arc<dyn antecedent_core::ProgressSink> =
            std::sync::Arc::new(Count(std::sync::Arc::clone(&calls)));
        Python::attach(|py| {
            let (observer, owner) = run(py, move || {
                let observer = bind_progress(Some(inner)).unwrap();
                let owner = ACTIVE
                    .with(|active| std::sync::Arc::downgrade(active.borrow().as_ref().unwrap()));
                observer.report(0.5, "active");
                Ok((observer, owner))
            })
            .unwrap();
            assert!(owner.upgrade().is_none());
            observer.report(1.0, "after completion");
            assert_eq!(calls.load(super::Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn a_domain_message_cannot_consume_another_live_runs_callback_tag() {
        use super::{callback_failure, run};
        use crate::CausalError;
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        let (tag_sent, tag_received) = std::sync::mpsc::channel();
        let (spoof_finished, owner_released) = std::sync::mpsc::channel();
        std::thread::scope(|threads| {
            let owner = threads.spawn(move || {
                Python::attach(|py| {
                    let expected = PyValueError::new_err("owned callback");
                    let original = expected.clone_ref(py);
                    let error = run::<_, ()>(py, move || {
                        let message = callback_failure("owner", original);
                        tag_sent.send(message.clone()).unwrap();
                        // This worker is detached from Python while the other run
                        // attempts attribution with the still-live owner's tag.
                        owner_released.recv().unwrap();
                        Err(CausalError::new_err(message))
                    })
                    .unwrap_err();
                    assert!(error.cause(py).unwrap().value(py).is(expected.value(py)));
                });
            });
            let spoofer = threads.spawn(move || {
                let message = tag_received.recv().unwrap();
                Python::attach(|py| {
                    let result = run::<_, ()>(py, move || {
                        Err(CausalError::new_err(format!("unrelated domain error: {message}")))
                    });
                    // The foreign tag stayed live throughout attribution. Release its
                    // owner before any assertion, so a regression fails without leaving
                    // the other native worker waiting for an assertion that panicked.
                    spoof_finished.send(()).unwrap();
                    let error = result.unwrap_err();
                    assert!(error.is_instance_of::<CausalError>(py));
                    assert!(error.cause(py).is_none());
                });
            });
            owner.join().unwrap();
            spoofer.join().unwrap();
        });
    }

    #[test]
    fn an_unretained_outer_failure_cannot_attribute_its_nested_callbacks_original() {
        use super::{callback_failure, run, unretained_callback_description};
        use crate::CausalError;
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            let id = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let recorded = std::sync::Arc::clone(&id);
            let error = run::<_, ()>(py, move || {
                let nested = callback_failure("nested formatting", PyValueError::new_err("inner"));
                recorded.store(tagged_ids(&nested).next().unwrap(), super::Ordering::Relaxed);
                // Exercise the exact capacity-refusal transformation, without filling
                // the process-global stash and competing with concurrently running tests.
                let description = format!("outer callback: {nested}");
                let message = unretained_callback_description(&description);
                assert!(message.contains("inner"));
                assert_eq!(tagged_ids(&message).count(), 0);
                Err(CausalError::new_err(message))
            })
            .unwrap_err();
            assert!(error.is_instance_of::<CausalError>(py));
            assert!(error.cause(py).is_none());
            assert!(!super::STASH.lock().unwrap().contains_key(&id.load(super::Ordering::Relaxed)));
        });
    }

    #[test]
    fn unscoped_callback_failures_have_no_unowned_retention_tag() {
        use pyo3::Python;

        Python::initialize();
        Python::attach(|_| {
            assert!(super::ACTIVE.with(|active| active.borrow().is_none()));
            let message = super::callback_failure(
                "unscoped callback",
                pyo3::exceptions::PyValueError::new_err("plain failure"),
            );
            assert!(message.contains("plain failure"));
            assert_eq!(tagged_ids(&message).count(), 0);
        });
    }

    #[pyo3::pyfunction]
    fn record_nested_failure() -> String {
        super::callback_failure(
            "nested exception formatting",
            pyo3::exceptions::PyValueError::new_err("nested"),
        )
    }

    #[test]
    fn exception_formatting_can_reenter_callback_failure_without_holding_the_stash_lock() {
        use super::{callback_failure, run};
        use crate::CausalError;
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                pyo3::ffi::c_str!("class ReentrantError(Exception):\n    def __str__(self):\n        return record_nested_failure()\n"),
                pyo3::ffi::c_str!("callback_reentry.py"),
                pyo3::ffi::c_str!("callback_reentry"),
            ).unwrap();
            module
                .add_function(pyo3::wrap_pyfunction!(record_nested_failure, &module).unwrap())
                .unwrap();
            let original =
                PyErr::from_value(module.getattr("ReentrantError").unwrap().call0().unwrap());
            let expected = original.clone_ref(py);
            let ids = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let recorded = std::sync::Arc::clone(&ids);
            let error = run::<_, ()>(py, move || {
                let message = callback_failure("callback", original);
                *recorded.lock().unwrap() = tagged_ids(&message).collect();
                Err(CausalError::new_err(message))
            })
            .unwrap_err();
            assert!(error.cause(py).unwrap().value(py).is(expected.value(py)));
            let ids = ids.lock().unwrap().clone();
            assert_eq!(ids.len(), 2);
            let stash = super::STASH.lock().unwrap();
            assert!(ids.iter().all(|id| !stash.contains_key(id)));
        });
    }

    #[test]
    fn every_live_pending_context_joins_the_next_run() {
        use super::{PENDING, link};
        use antecedent_core::CancellationToken;
        PENDING.with(|pending| pending.borrow_mut().clear());
        let live: Vec<CancellationToken> = (0..500).map(|_| CancellationToken::new()).collect();
        for token in &live {
            link(token);
        }
        let queued = PENDING.with(|pending| {
            pending
                .borrow()
                .iter()
                .filter_map(antecedent_core::WeakCancellationToken::upgrade)
                .count()
        });
        assert_eq!(queued, live.len());
        PENDING.with(|pending| pending.borrow_mut().clear());
    }

    #[test]
    fn tokens_of_dropped_contexts_do_not_accumulate() {
        use super::{PENDING, link};
        use antecedent_core::CancellationToken;
        PENDING.with(|pending| pending.borrow_mut().clear());
        for _ in 0..10_000 {
            link(&CancellationToken::new());
        }
        assert!(PENDING.with(|pending| pending.borrow().len()) < 200);
        PENDING.with(|pending| pending.borrow_mut().clear());
    }
}
