//! Shared fit ownership and cancellation for prepared policy execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use antecedent::prelude::ExecutionContext;
use pyo3::prelude::*;

use crate::py_err;

// A single successful fit is shared across callers. The physical mutex protects
// only state transitions; no computation or Python observer runs under it.
enum FitState<T> {
    Empty,
    Fitting { observer: bool },
    Ready(Arc<T>),
}

pub(super) struct FitCache<T> {
    state: Mutex<FitState<T>>,
    pub(super) published: AtomicBool,
}

impl<T> Default for FitCache<T> {
    fn default() -> Self {
        Self { state: Mutex::new(FitState::Empty), published: AtomicBool::new(false) }
    }
}

struct FitOwner<'a, T>(&'a FitCache<T>);
impl<T> Drop for FitOwner<'_, T> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if matches!(*state, FitState::Fitting { .. }) {
            *state = FitState::Empty;
        }
    }
}

impl<T> FitCache<T> {
    pub(super) fn materialize(
        &self,
        ctx: &ExecutionContext,
        fit: impl FnOnce() -> PyResult<T>,
        has_observer: bool,
        mut observe: impl FnMut(f64),
        mut on_wait: impl FnMut(),
    ) -> PyResult<Arc<T>> {
        let cancelled = || py_err(antecedent::CausalError::Cancelled { stage: "policy_fit" });
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            match &*state {
                FitState::Ready(value) => return Ok(Arc::clone(value)),
                FitState::Empty => {
                    *state = FitState::Fitting { observer: false };
                    break;
                }
                FitState::Fitting { observer: true } => {
                    return Err(crate::CausalStateError::new_err(
                        "policy fit is busy invoking a progress observer; retry after it returns",
                    ));
                }
                FitState::Fitting { observer: false } => {}
            }
            drop(state);
            on_wait();
            std::thread::park_timeout(std::time::Duration::from_millis(5));
        }
        let _owner = FitOwner(self);
        let mut report = |fraction| -> PyResult<()> {
            if !has_observer {
                return if ctx.cancellation.is_cancelled() { Err(cancelled()) } else { Ok(()) };
            }
            {
                let mut state =
                    self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                *state = FitState::Fitting { observer: true };
            }
            observe(fraction);
            {
                let mut state =
                    self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                *state = FitState::Fitting { observer: false };
            }
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            Ok(())
        };
        report(0.0)?;
        let value = Arc::new(fit()?);
        report(1.0)?;
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = FitState::Ready(Arc::clone(&value));
        self.published.store(true, Ordering::Release);
        Ok(value)
    }
}
