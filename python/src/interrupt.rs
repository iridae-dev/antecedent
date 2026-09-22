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
//! interrupt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
const MAX_STASHED: usize = 256;

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

/// Callback failures awaiting attribution, by the id their message carries.
static STASH: Mutex<BTreeMap<u64, PyErr>> = Mutex::new(BTreeMap::new());

/// Source of callback-failure ids; one id per recorded failure, never reused.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

const TAG_OPEN: &str = "[callback failure #";

/// Record the Python exception behind a callback failure and return the message the domain error
/// should carry. The message ends in an id tag that [`attribute`] looks the exception up by, so
/// two runs whose callbacks fail with byte-identical text can never be attributed to each other.
pub(crate) fn callback_failure(what: &str, err: PyErr) -> String {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let message = format!("{what}: {err} {TAG_OPEN}{id}]");
    let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
    while stash.len() >= MAX_STASHED {
        stash.pop_first();
    }
    stash.insert(id, err);
    message
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
pub(crate) fn attribute(py: Python<'_>, err: PyErr) -> PyErr {
    let text = err.value(py).to_string();
    let mut stash = STASH.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(original) = tagged_ids(&text).find_map(|id| stash.remove(&id)) else {
        return err;
    };
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
}
