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
//! to a message. [`callback_failure`] keeps the original exception in the executing run's own
//! stash and returns a structured [`CallbackFailure`] that the native error enums carry as a
//! variant (`ModelError::CallbackFailed`, `StatsError::CallbackFailed`, ...). Converting such a
//! native error to its Python exception ([`with_callback_failure`]) records the failure's id on
//! the exception; [`attribute`] looks that id up in the run's stash and re-raises the original
//! (`KeyboardInterrupt`, `SystemExit`, ... unchanged) or chains it as `__cause__` of the domain
//! error, so tracebacks into user code survive and `except CausalError` cannot swallow an
//! interrupt. No message text is ever parsed. Each native run owns its stash, bounded by the
//! run policy's callback-failure capacity, and releases any unattributed failures when the run
//! finishes, including successful fallbacks and signal overrides.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::Duration;

use antecedent_core::{CallbackFailure, CancellationToken, WeakCancellationToken};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::{CausalError, catch_ffi};

/// How often the calling thread re-acquires the GIL to look for a pending signal.
const POLL: Duration = Duration::from_millis(50);

/// Native analyses recurse (graph search, tree builders); the worker gets a stack no smaller than
/// the 8 MiB a Python main thread has by default, with headroom for the engines' own frames.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Callback failures one run retains when its execution policy names no capacity. A full
/// stash refuses a new id instead of dropping the oldest, which may still be waiting to be
/// attributed.
pub(crate) const DEFAULT_CALLBACK_FAILURE_CAPACITY: usize = 256;

/// Tokens the interrupt loop cancels for one run.
struct Scope {
    own: CancellationToken,
    linked: Mutex<Vec<CancellationToken>>,
    /// Python exceptions retained by callbacks on this worker, by the id their
    /// [`CallbackFailure`] carries, including errors a native fallback suppresses or a
    /// pending signal supersedes before attribution. Owned by this run alone, so only the
    /// run's own ids are ever eligible for attribution.
    failures: Mutex<BTreeMap<u64, PyErr>>,
    /// Most failures `failures` retains; a further failure is reported without an id.
    capacity: usize,
    /// First observer exception or callback BaseException. Native success/fallback
    /// must not swallow an observer failure or a request to interrupt or exit.
    /// Bounded independently of `failures`.
    interruption: Mutex<Option<PyErr>>,
}

impl Scope {
    fn new(capacity: usize) -> Self {
        Self {
            own: CancellationToken::default(),
            linked: Mutex::new(Vec::new()),
            failures: Mutex::new(BTreeMap::new()),
            capacity,
            interruption: Mutex::new(None),
        }
    }

    fn cancel_all(&self) {
        self.own.cancel();
        for token in self.linked.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            token.cancel();
        }
    }

    fn release_failures(&self) {
        // Never release Python exceptions while holding the stash mutex: their
        // destruction may execute Python finalizers and re-enter a callback.
        let discarded =
            std::mem::take(&mut *self.failures.lock().unwrap_or_else(PoisonError::into_inner));
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

struct ScopedStage {
    inner: Arc<dyn antecedent::StageResultSink>,
    scope: std::sync::Weak<Scope>,
}

impl antecedent::StageResultSink for ScopedStage {
    fn on_stage(&self, event: &antecedent::StageEvent) {
        let Some(scope) = self.scope.upgrade() else { return };
        let _restore = ActiveGuard(ACTIVE.with(|active| active.replace(Some(scope))));
        self.inner.on_stage(event);
    }
}

/// Bind a click-only stage observer inside its active run, including callbacks
/// dispatched by native pool workers. Retaining a sink never retains that run.
pub(crate) fn bind_stage_sink(
    inner: Arc<dyn antecedent::StageResultSink>,
) -> Arc<dyn antecedent::StageResultSink> {
    let scope = ACTIVE.with(|active| active.borrow().as_ref().map(Arc::downgrade));
    match scope {
        Some(scope) => Arc::new(ScopedStage { inner, scope }),
        None => inner,
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

/// An execution-local callback owner. Weak ownership prevents a retained adapter
/// from keeping a completed run alive or attributing a late callback to a new run.
struct ValidatorScope(Weak<Scope>);

impl ValidatorScope {
    fn call<T>(&self, callback: impl FnOnce() -> T) -> Option<T> {
        let scope = self.0.upgrade()?;
        let _restore = ActiveGuard(ACTIVE.with(|active| active.replace(Some(scope))));
        Some(callback())
    }
}

struct ScopedValidator {
    inner: Arc<dyn antecedent_validate::CustomEffectValidator>,
    scope: ValidatorScope,
}

impl antecedent_validate::CustomEffectValidator for ScopedValidator {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn validate(
        &self,
        problem: &antecedent_validate::RefutationProblem<'_>,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<antecedent_validate::RefutationReport, antecedent_validate::ValidationError> {
        self.scope.call(|| self.inner.validate(problem, ctx)).unwrap_or_else(|| {
            Err(antecedent_validate::ValidationError::data_msg(
                "custom validator execution scope has ended",
            ))
        })
    }
}

/// Bind validators only around an execution, never on a retained preparation.
/// Native snapshots, scientific plans, and callback objects remain shared; only
/// the small execution backend is cloned when callbacks actually exist.
/// Callers retain the original study, not the temporary weak adapters.
pub(crate) fn bind_study_validators(
    study: &antecedent::PreparedStudy,
) -> PyResult<std::borrow::Cow<'_, antecedent::PreparedStudy>> {
    if study.custom_validators().is_empty() {
        return Ok(std::borrow::Cow::Borrowed(study));
    }
    let scope = ACTIVE
        .with(|active| active.borrow().as_ref().map(Arc::downgrade))
        .ok_or_else(|| crate::value_err("custom validators require an active native execution"))?;
    let validators = study
        .custom_validators()
        .iter()
        .map(|inner| {
            Arc::new(ScopedValidator {
                inner: Arc::clone(inner),
                scope: ValidatorScope(scope.clone()),
            }) as Arc<dyn antecedent_validate::CustomEffectValidator>
        })
        .collect();
    let mut execution = study.clone();
    execution.rebind_custom_validators(validators).map_err(crate::py_err)?;
    Ok(std::borrow::Cow::Owned(execution))
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

/// Source of callback-failure ids; one id per recorded failure, never reused.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The attribute that carries a retained failure's id on the Python exception a native error
/// was converted to.
const FAILURE_ID_ATTRIBUTE: &str = "_antecedent_callback_failure";

/// Record the Python exception behind a callback failure and return it as structured data.
/// An ordinary exception is retained in the executing run's stash under the returned
/// failure's id, for [`attribute`] to find through [`with_callback_failure`], so concurrent
/// runs cannot misattribute a traceback. A scoped BaseException cancels the run and is
/// retained directly, even if native code converts its error into a report. A failure no run
/// can retain (no executing run, or a full stash) carries no id.
pub(crate) fn callback_failure(what: &str, err: PyErr) -> CallbackFailure {
    let scope = ACTIVE.with(|active| active.borrow().clone());
    // No run owns an unscoped failure, so there is no later attribution or cleanup
    // boundary at which retaining its Python exception would be safe.
    let Some(scope) = scope else {
        return described(None, format!("{what}: {err}"));
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
            return described(None, format!("{what}: Python callback interrupted the native run"));
        }
    }
    // Exception __str__ can execute user Python, including another failing
    // callback. Format ordinary errors before locking, even when the stash is full.
    let description = format!("{what}: {err}");
    let mut stash = scope.failures.lock().unwrap_or_else(PoisonError::into_inner);
    if stash.len() >= scope.capacity {
        // The oldest entry may still be waiting for [`attribute`]. Dropping it
        // would attach this failure's exception to an earlier failure. The new
        // failure stays in the message and is not given an id.
        drop(stash);
        return described(None, description);
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    stash.insert(id, err);
    drop(stash);
    described(Some(id), description)
}

/// A failure with its description, naming the registered reason code the foreign error's own
/// text opens with (directly or after one `<label>: `), if it names one.
fn described(id: Option<u64>, description: String) -> CallbackFailure {
    let code = crate::errors::reason_code_in_message(&description)
        .filter(|code| antecedent_core::reason_code::is_registered(code))
        .map(str::to_owned);
    CallbackFailure::new(id, description, code)
}

/// Record on a converted Python exception which retained failure it stands for, so the run
/// boundary can restore the original exception as its cause.
pub(crate) fn with_callback_failure(err: PyErr, failure: &CallbackFailure) -> PyErr {
    if let Some(id) = failure.id() {
        Python::attach(|py| {
            let _ = err.value(py).setattr(FAILURE_ID_ATTRIBUTE, id);
        });
    }
    err
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

/// Replace a domain error that stands for a callback failure with the Python exception behind it.
///
/// A non-`Exception` (`KeyboardInterrupt`, `SystemExit`, ...) is re-raised as is. An `Exception`
/// becomes the `__cause__` of the domain error, keeping its type and traceback.
/// Only this run's own retained failures are eligible: an id the run's stash does not hold
/// (another run's, or one a user exception merely claims) attributes nothing.
fn attribute(py: Python<'_>, err: PyErr, scope: &Scope) -> PyErr {
    if !err.is_instance_of::<PyException>(py) {
        return err;
    }
    // Reading the attribute can call Python; do it before taking the stash lock.
    let id =
        err.value(py).getattr(FAILURE_ID_ATTRIBUTE).ok().and_then(|id| id.extract::<u64>().ok());
    let Some(id) = id else { return err };
    let original = scope.failures.lock().unwrap_or_else(PoisonError::into_inner).remove(&id);
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
    let scope = Arc::new(Scope::new(crate::execution_api::callback_failure_capacity()));
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
#[allow(
    clippy::disallowed_methods,
    reason = "the tests raise Python exceptions from callbacks to check how the run attributes them"
)]
mod tests {
    use super::{DEFAULT_CALLBACK_FAILURE_CAPACITY, Scope, with_callback_failure};
    use antecedent_core::CallbackFailure;

    #[test]
    fn a_domain_error_names_its_callback_failure_by_id_not_by_text() {
        use crate::CausalError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            // Two failures with byte-identical text carry different ids, and the id rides
            // on the converted exception, never in its message.
            let first = CallbackFailure::new(Some(7), "Python CI callback failed: boom", None);
            let second = CallbackFailure::new(Some(8), "Python CI callback failed: boom", None);
            assert_eq!(first.description(), second.description());
            let converted = with_callback_failure(CausalError::new_err(first.to_string()), &second);
            assert_eq!(converted.value(py).to_string(), "Python CI callback failed: boom");
            let id: u64 = converted
                .value(py)
                .getattr(super::FAILURE_ID_ATTRIBUTE)
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(id, 8);
            // A failure no run retained names nothing.
            let unretained = CallbackFailure::new(None, "boom", None);
            let plain = with_callback_failure(CausalError::new_err("boom"), &unretained);
            assert!(plain.value(py).getattr(super::FAILURE_ID_ATTRIBUTE).is_err());
        });
    }

    #[test]
    fn a_full_callback_stash_does_not_drop_the_oldest_id() {
        use super::{ACTIVE, callback_failure};
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|_| {
            let scope = std::sync::Arc::new(Scope::new(2));
            let _active = super::ActiveGuard(ACTIVE.with(|a| a.replace(Some(scope.clone()))));
            let first = callback_failure("cb", PyValueError::new_err("one"));
            let second = callback_failure("cb", PyValueError::new_err("two"));
            let refused = callback_failure("cb", PyValueError::new_err("three"));
            assert!(first.id().is_some() && second.id().is_some());
            assert_eq!(refused.id(), None, "a full stash refuses instead of evicting");
            assert!(refused.description().contains("three"));
            let held: Vec<u64> = scope.failures.lock().unwrap().keys().copied().collect();
            assert_eq!(held, vec![first.id().unwrap(), second.id().unwrap()]);
        });
    }

    #[test]
    fn the_callback_failure_capacity_defaults_and_follows_the_run_policy() {
        assert_eq!(DEFAULT_CALLBACK_FAILURE_CAPACITY, 256);
        assert_eq!(
            crate::execution_api::callback_failure_capacity(),
            DEFAULT_CALLBACK_FAILURE_CAPACITY
        );
    }

    #[test]
    fn successful_fallbacks_release_callback_exceptions_after_every_run() {
        use super::{callback_failure, run};
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for _ in 0..=DEFAULT_CALLBACK_FAILURE_CAPACITY {
                let retained = std::sync::Arc::new(std::sync::Mutex::new(None));
                let worker_retained = std::sync::Arc::clone(&retained);
                let id = run(py, || {
                    // Cleanup belongs to the completed run, even when a native
                    // owner still holds its scope after the worker returns.
                    *worker_retained.lock().unwrap() =
                        super::ACTIVE.with(|active| active.borrow().clone());
                    let failure =
                        callback_failure("suppressed callback", PyValueError::new_err("boom"));
                    Ok(failure.id().expect("a run's stash always has room at its start"))
                })
                .unwrap();
                let scope = retained.lock().unwrap().clone().expect("the worker's scope");
                assert!(!scope.failures.lock().unwrap().contains_key(&id));
            }
        });
    }

    #[test]
    fn callback_base_exceptions_survive_and_superseded_failures_are_released() {
        use super::{callback_failure, run, with_callback_failure};
        use crate::CausalError;
        use pyo3::exceptions::{PyKeyboardInterrupt, PySystemExit, PyValueError};
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            for original in [PyKeyboardInterrupt::new_err("stop"), PySystemExit::new_err(17)] {
                let expected = original.clone_ref(py);
                let error = run::<_, ()>(py, move || {
                    let failure = callback_failure("callback", original);
                    Err(with_callback_failure(CausalError::new_err(failure.to_string()), &failure))
                })
                .unwrap_err();
                assert!(error.value(py).is(expected.value(py)));
            }
            let retained = std::sync::Arc::new(std::sync::Mutex::new(None));
            let worker_retained = std::sync::Arc::clone(&retained);
            let error = run::<_, ()>(py, move || {
                *worker_retained.lock().unwrap() =
                    super::ACTIVE.with(|active| active.borrow().clone());
                let failure = callback_failure("discarded callback", PyValueError::new_err("boom"));
                assert!(failure.id().is_some());
                // Mirrors run's signal-wins branch: this result carries no callback id.
                Err(PyKeyboardInterrupt::new_err("signal wins"))
            })
            .unwrap_err();
            assert!(error.is_instance_of::<PyKeyboardInterrupt>(py));
            let scope = retained.lock().unwrap().clone().expect("the worker's scope");
            assert!(scope.failures.lock().unwrap().is_empty());
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
                                Err(with_callback_failure(
                                    CausalError::new_err(message.to_string()),
                                    &message,
                                ))
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
    fn native_validator_workers_preserve_concurrent_originals_after_diagnostic_conversion() {
        use super::{ACTIVE, ValidatorScope, observer_failure, run};
        use pyo3::exceptions::{PyRuntimeError, PySystemExit};
        use pyo3::prelude::*;
        use pyo3::types::PyDict;
        use std::sync::Arc;

        Python::initialize();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|threads| {
            let callers: Vec<_> = (0..2)
                .map(|caller| {
                    let barrier = Arc::clone(&barrier);
                    threads.spawn(move || {
                        Python::attach(|py| {
                            let globals = PyDict::new(py);
                            let original = if caller == 0 {
                                PyRuntimeError::new_err("identical validator failure")
                            } else {
                                PySystemExit::new_err("identical validator failure")
                            };
                            globals.set_item("original", original.value(py)).unwrap();
                            py.run(
                                pyo3::ffi::c_str!("def validate():\n    raise original\n"),
                                Some(&globals),
                                None,
                            )
                            .unwrap();
                            let callback = globals.get_item("validate").unwrap().unwrap().unbind();
                            // Reuse the same Python callback across fresh execution scopes.
                            for execution in 0..2 {
                                let callback = callback.clone_ref(py);
                                let barrier = Arc::clone(&barrier);
                                let error = run::<_, ()>(py, move || {
                                    let scope = ValidatorScope(ACTIVE.with(|active| {
                                        Arc::downgrade(active.borrow().as_ref().unwrap())
                                    }));
                                    let ctx = crate::py_execution_context(1, 2);
                                    if execution == 0 {
                                        barrier.wait();
                                    }
                                    ctx.map_indexed(2, |_, _| {
                                        assert!(ACTIVE.with(|active| active.borrow().is_none()));
                                        let diagnostic = scope
                                            .call(|| {
                                                Python::attach(|py| {
                                                    let error =
                                                        callback.bind(py).call0().unwrap_err();
                                                    observer_failure(error).unwrap();
                                                    // Same conversion performed by PyCustomValidator;
                                                    // a native validation diagnostic must not own the error.
                                                    antecedent_validate::ValidationError::data_msg(
                                                        "Python validator raised an exception",
                                                    )
                                                })
                                            })
                                            .unwrap();
                                        assert!(diagnostic.to_string().contains("validator"));
                                        assert!(ACTIVE.with(|active| active.borrow().is_none()));
                                        Ok::<(), ()>(())
                                    })
                                    .unwrap();
                                    Ok(()) // Native computation-failure recording or fallback succeeded.
                                })
                                .unwrap_err();
                                assert!(error.value(py).is(original.value(py)));
                                assert!(
                                    error
                                        .value(py)
                                        .getattr("_antecedent_operational_callback")
                                        .unwrap()
                                        .extract::<bool>()
                                        .unwrap()
                                );
                                let mut traceback =
                                    error.value(py).getattr("__traceback__").unwrap();
                                let mut saw_validator = false;
                                while !traceback.is_none() {
                                    let function: String = traceback
                                        .getattr("tb_frame")
                                        .unwrap()
                                        .getattr("f_code")
                                        .unwrap()
                                        .getattr("co_name")
                                        .unwrap()
                                        .extract()
                                        .unwrap();
                                    saw_validator |= function == "validate";
                                    traceback = traceback.getattr("tb_next").unwrap();
                                }
                                assert!(saw_validator);
                            }
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
    fn validator_scope_restores_tls_on_panic_and_cannot_leak_into_later_runs() {
        use super::{ACTIVE, ValidatorScope, run};
        use pyo3::prelude::*;
        use std::sync::Arc;

        Python::initialize();
        Python::attach(|py| {
            let adapter = run(py, || {
                let owner = ACTIVE.with(|active| Arc::downgrade(active.borrow().as_ref().unwrap()));
                let adapter = ValidatorScope(owner);
                let expected = ACTIVE.with(|active| active.borrow().clone().unwrap());
                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    adapter.call(|| panic!("validator unwound"));
                }));
                assert!(unwind.is_err());
                assert!(
                    ACTIVE
                        .with(|active| Arc::ptr_eq(active.borrow().as_ref().unwrap(), &expected,))
                );
                Ok(adapter)
            })
            .unwrap();
            assert!(adapter.0.upgrade().is_none(), "validator must not retain its run");
            assert!(adapter.call(|| panic!("late callback must not execute")).is_none());
            run(py, || {
                assert!(!super::ambient_token().unwrap().is_cancelled());
                // The next run's active TLS cannot revive a prior adapter.
                assert!(adapter.call(|| panic!("must not borrow next run")).is_none());
                Ok(())
            })
            .unwrap();
        });
    }

    #[test]
    fn native_stage_workers_preserve_concurrent_original_exceptions_and_tracebacks() {
        use super::{ambient_token, bind_stage_sink, run};
        use pyo3::exceptions::{PyRuntimeError, PySystemExit};
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        Python::initialize();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|threads| {
            let callers: Vec<_> = (0..2)
                .map(|caller| {
                    let barrier = std::sync::Arc::clone(&barrier);
                    threads.spawn(move || {
                        Python::attach(|py| {
                            let globals = PyDict::new(py);
                            let expected = if caller == 0 {
                                PyRuntimeError::new_err("identical stage failure")
                            } else {
                                PySystemExit::new_err("identical stage failure")
                            };
                            globals.set_item("original", expected.value(py)).unwrap();
                            py.run(
                                pyo3::ffi::c_str!("def observe(*args):\n    raise original\n"),
                                Some(&globals),
                                None,
                            )
                            .unwrap();
                            let observer: std::sync::Arc<dyn antecedent::StageResultSink> =
                                std::sync::Arc::new(crate::callbacks::PyStageResultSink::new(
                                    globals.get_item("observe").unwrap().unwrap().unbind(),
                                ));
                            // Raw callback reuse must bind to each new execution's scope.
                            for execution in 0..2 {
                                let observer = std::sync::Arc::clone(&observer);
                                let barrier = std::sync::Arc::clone(&barrier);
                                let error = run::<_, ()>(py, move || {
                                    let ctx = crate::py_execution_context(1, 2);
                                    let observer = bind_stage_sink(observer);
                                    let owner = std::thread::current().id();
                                    if execution == 0 {
                                        barrier.wait();
                                    }
                                    ctx.map_indexed(2, |_, _| {
                                        assert_ne!(owner, std::thread::current().id());
                                        assert!(ambient_token().is_none());
                                        observer.on_stage(&antecedent::StageEvent::Validate {
                                            refutations: Vec::new(),
                                            predictive_checks: Vec::new(),
                                        });
                                        assert!(ambient_token().is_none());
                                        Ok::<(), ()>(())
                                    })
                                    .unwrap();
                                    // Even native success must not swallow observer failure.
                                    Ok(())
                                })
                                .unwrap_err();
                                assert!(error.value(py).is(expected.value(py)));
                                let traceback = error.value(py).getattr("__traceback__").unwrap();
                                let function: String = traceback
                                    .getattr("tb_frame")
                                    .unwrap()
                                    .getattr("f_code")
                                    .unwrap()
                                    .getattr("co_name")
                                    .unwrap()
                                    .extract()
                                    .unwrap();
                                assert_eq!(function, "observe");
                            }
                            globals.clear();
                        });
                    })
                })
                .collect();
            for caller in callers {
                caller.join().unwrap();
            }
        });
    }

    #[test]
    fn retained_stage_adapter_releases_its_run_and_ignores_late_events() {
        use super::{ACTIVE, bind_stage_sink, run};
        use pyo3::prelude::*;

        struct Count(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl antecedent::StageResultSink for Count {
            fn on_stage(&self, _: &antecedent::StageEvent) {
                self.0.fetch_add(1, super::Ordering::Relaxed);
            }
        }
        Python::initialize();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let inner: std::sync::Arc<dyn antecedent::StageResultSink> =
            std::sync::Arc::new(Count(std::sync::Arc::clone(&calls)));
        Python::attach(|py| {
            let event = antecedent::StageEvent::Validate {
                refutations: Vec::new(),
                predictive_checks: Vec::new(),
            };
            let (observer, owner) = run(py, move || {
                let observer = bind_stage_sink(inner);
                let owner = ACTIVE
                    .with(|active| std::sync::Arc::downgrade(active.borrow().as_ref().unwrap()));
                observer.on_stage(&antecedent::StageEvent::Validate {
                    refutations: Vec::new(),
                    predictive_checks: Vec::new(),
                });
                Ok((observer, owner))
            })
            .unwrap();
            assert!(owner.upgrade().is_none(), "a retained sink must not retain its run");
            observer.on_stage(&event);
            assert_eq!(calls.load(super::Ordering::Relaxed), 1);
            // A following execution on the same caller starts without cancellation.
            run(py, || {
                assert!(!super::ambient_token().unwrap().is_cancelled());
                Ok(())
            })
            .unwrap();
        });
    }

    #[test]
    fn a_domain_error_cannot_consume_another_live_runs_callback_failure() {
        use super::{callback_failure, run, with_callback_failure};
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
                        let failure = callback_failure("owner", original);
                        tag_sent.send(failure.clone()).unwrap();
                        // This worker is detached from Python while the other run
                        // attempts attribution with the still-live owner's id.
                        owner_released.recv().unwrap();
                        Err(with_callback_failure(
                            CausalError::new_err(failure.to_string()),
                            &failure,
                        ))
                    })
                    .unwrap_err();
                    assert!(error.cause(py).unwrap().value(py).is(expected.value(py)));
                });
            });
            let spoofer = threads.spawn(move || {
                let foreign = tag_received.recv().unwrap();
                Python::attach(|py| {
                    let result = run::<_, ()>(py, move || {
                        // An unrelated error that names the other run's failure id.
                        Err(with_callback_failure(
                            CausalError::new_err("unrelated domain error"),
                            &foreign,
                        ))
                    });
                    // The foreign failure stayed live throughout attribution. Release its
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
        use super::{callback_failure, run, with_callback_failure};
        use crate::CausalError;
        use pyo3::exceptions::PyValueError;
        use pyo3::prelude::*;

        Python::initialize();
        Python::attach(|py| {
            let retained = std::sync::Arc::new(std::sync::Mutex::new(None));
            let worker_retained = std::sync::Arc::clone(&retained);
            let error = run::<_, ()>(py, move || {
                *worker_retained.lock().unwrap() =
                    super::ACTIVE.with(|active| active.borrow().clone());
                let nested = callback_failure("nested formatting", PyValueError::new_err("inner"));
                assert!(nested.id().is_some());
                // An outer failure that could not retain its original (a full stash) is an
                // id-less failure; its description may quote the nested one, but only the
                // outer failure's own (absent) id is ever recorded on the converted error.
                let outer = antecedent_core::CallbackFailure::new(
                    None,
                    format!("outer callback: {nested}"),
                    None,
                );
                assert!(outer.description().contains("inner"));
                Err(with_callback_failure(CausalError::new_err(outer.to_string()), &outer))
            })
            .unwrap_err();
            assert!(error.is_instance_of::<CausalError>(py));
            assert!(error.cause(py).is_none());
            let scope = retained.lock().unwrap().clone().expect("the worker's scope");
            assert!(scope.failures.lock().unwrap().is_empty());
        });
    }

    #[test]
    fn unscoped_callback_failures_are_not_retained() {
        use pyo3::Python;

        Python::initialize();
        Python::attach(|_| {
            assert!(super::ACTIVE.with(|active| active.borrow().is_none()));
            let failure = super::callback_failure(
                "unscoped callback",
                pyo3::exceptions::PyValueError::new_err("plain failure"),
            );
            assert!(failure.description().contains("plain failure"));
            assert_eq!(failure.id(), None);
        });
    }

    #[pyo3::pyfunction]
    fn record_nested_failure() -> String {
        super::callback_failure(
            "nested exception formatting",
            pyo3::exceptions::PyValueError::new_err("nested"),
        )
        .to_string()
    }

    #[test]
    fn exception_formatting_can_reenter_callback_failure_without_holding_the_stash_lock() {
        use super::{callback_failure, run, with_callback_failure};
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
            let retained = std::sync::Arc::new(std::sync::Mutex::new(None));
            let worker_retained = std::sync::Arc::clone(&retained);
            let error = run::<_, ()>(py, move || {
                let scope = super::ACTIVE.with(|active| active.borrow().clone()).unwrap();
                let failure = callback_failure("callback", original);
                // The exception's own formatting registered a nested failure too.
                assert_eq!(scope.failures.lock().unwrap().len(), 2);
                *worker_retained.lock().unwrap() = Some(scope);
                Err(with_callback_failure(CausalError::new_err(failure.to_string()), &failure))
            })
            .unwrap_err();
            assert!(error.cause(py).unwrap().value(py).is(expected.value(py)));
            let scope = retained.lock().unwrap().clone().expect("the worker's scope");
            assert!(scope.failures.lock().unwrap().is_empty());
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
