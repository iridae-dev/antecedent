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
//! to a message. [`callback_failure`] keeps the original exception; [`attribute`] re-raises it
//! (`KeyboardInterrupt`, `SystemExit`, ... unchanged) or chains it as `__cause__` of the domain
//! error, so tracebacks into user code survive and `except CausalError` cannot swallow an
//! interrupt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use antecedent_core::CancellationToken;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::{CausalError, catch_ffi};

/// How often the calling thread re-acquires the GIL to look for a pending signal.
const POLL: Duration = Duration::from_millis(50);

/// Native analyses recurse (graph search, tree builders); the worker gets a stack no smaller than
/// the 8 MiB a Python main thread has by default, with headroom for the engines' own frames.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Retained callback failures; older entries are stale (their run already returned).
const MAX_STASHED: usize = 16;

/// Tokens the interrupt loop cancels for one run.
#[derive(Default)]
struct Scope {
    own: CancellationToken,
    linked: Mutex<Vec<CancellationToken>>,
}

impl Scope {
    fn cancel_all(&self) {
        self.own.cancel();
        for token in self.linked.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            token.cancel();
        }
    }
}

/// Bound on tokens waiting for the next run on a thread (contexts built and never run).
const MAX_PENDING: usize = 64;

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Scope>>> = const { RefCell::new(None) };
    /// Tokens of contexts built on a Python thread ahead of the [`run`] that will execute them.
    static PENDING: RefCell<Vec<CancellationToken>> = const { RefCell::new(Vec::new()) };
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
            if pending.len() >= MAX_PENDING {
                pending.remove(0);
            }
            pending.push(token.clone());
        });
    }
}

static STASH: Mutex<Vec<(String, PyErr)>> = Mutex::new(Vec::new());

/// Record the Python exception behind a callback failure and return the message the domain error
/// should carry. The message doubles as the key [`attribute`] matches on.
pub(crate) fn callback_failure(what: &str, err: PyErr) -> String {
    let message = format!("{what}: {err}");
    let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
    if stash.len() >= MAX_STASHED {
        stash.remove(0);
    }
    stash.push((message.clone(), err));
    message
}

/// Replace a domain error that wraps a callback failure with the Python exception behind it.
///
/// A non-`Exception` (`KeyboardInterrupt`, `SystemExit`, ...) is re-raised as is. An `Exception`
/// becomes the `__cause__` of the domain error, keeping its type and traceback.
pub(crate) fn attribute(py: Python<'_>, err: PyErr) -> PyErr {
    let text = err.value(py).to_string();
    let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(at) = stash.iter().position(|(message, _)| text.contains(message.as_str())) else {
        return err;
    };
    let (_, original) = stash.remove(at);
    drop(stash);
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
/// result: the user asked for the run to stop.
pub(crate) fn run<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    let scope = Arc::new(Scope::default());
    scope
        .linked
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .extend(PENDING.with(|pending| std::mem::take(&mut *pending.borrow_mut())));
    let slot: (Mutex<Option<PyResult<T>>>, Condvar) = (Mutex::new(None), Condvar::new());
    let outcome = std::thread::scope(|s| {
        let worker_scope = Arc::clone(&scope);
        let slot = &slot;
        let spawned =
            std::thread::Builder::new().stack_size(WORKER_STACK_BYTES).spawn_scoped(s, move || {
                ACTIVE.with(|active| *active.borrow_mut() = Some(worker_scope));
                let result = catch_ffi(f);
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
        match interrupt {
            Some(e) => Err(e),
            None => result,
        }
    });
    outcome.map_err(|e| attribute(py, e))
}
