//! Explicit per-operation Python execution policy and scoped worker propagation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::sync::Arc;

use antecedent_core::{CachePolicy, ExecutionContext, KernelPolicy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub(crate) struct Policy {
    context: ExecutionContext,
    progress: Option<Arc<dyn antecedent_core::ProgressSink>>,
    cancel: Option<antecedent_core::CancellationToken>,
    /// Most Python callback failures one run retains for attribution.
    callback_failure_capacity: usize,
    /// Observed treatment ranges a pulse's active level may lie beyond the observed treatment
    /// interval; `None` keeps the engine default.
    pulse_extrapolation_ranges: Option<f64>,
}

thread_local! {
    static POLICY: RefCell<Option<Arc<Policy>>> = const { RefCell::new(None) };
}

pub(crate) fn capture() -> Option<Arc<Policy>> {
    POLICY.with(|policy| policy.borrow().clone())
}

/// The callback-failure capacity of the run being started on this thread: the explicit
/// execution's, else the default.
pub(crate) fn callback_failure_capacity() -> usize {
    capture().map_or(crate::interrupt::DEFAULT_CALLBACK_FAILURE_CAPACITY, |policy| {
        policy.callback_failure_capacity
    })
}

pub(crate) struct Guard(Option<Arc<Policy>>);

impl Drop for Guard {
    fn drop(&mut self) {
        POLICY.with(|policy| *policy.borrow_mut() = self.0.take());
    }
}

pub(crate) fn enter(policy: Option<Arc<Policy>>) -> Guard {
    Guard(POLICY.with(|current| current.replace(policy)))
}

/// Apply the exact caller policy captured at this explicit operation's boundary.
/// The scientific request's RNG and lowered-dispatch identity remain on ctx.
pub(crate) fn apply(ctx: &mut ExecutionContext) {
    let Some(policy) = capture() else { return };
    ctx.parallelism = policy.context.parallelism;
    ctx.admission = policy.context.admission.clone();
    ctx.cache_policy = policy.context.cache_policy;
    ctx.kernel_policy = policy.context.kernel_policy;
    if let Some(ranges) = policy.pulse_extrapolation_ranges {
        if let Some(scoped) = ctx.with_pulse_extrapolation_ranges(ranges) {
            *ctx = scoped;
        }
    }
    if let Some(progress) = &policy.progress {
        ctx.progress = crate::interrupt::bind_progress(Some(Arc::clone(progress)));
    }
    // Link every run to the explicit token, but do not let a run's cancellation
    // replace a caller-owned pool or be inferred from unrelated active runs.
    if let Some(cancel) = &policy.cancel {
        crate::install_cancellation(ctx, Some(cancel.clone()));
    }
}

#[pyclass(name = "_Execution", frozen)]
pub(crate) struct PythonExecution {
    policy: Arc<Policy>,
}

#[pymethods]
impl PythonExecution {
    #[new]
    #[pyo3(signature = (*, seed=1, threads=None, memory_bytes=None, cache="bounded", cache_bytes=Some(crate::PY_DEFAULT_CACHE_MAX_BYTES), kernel="auto", cancel=None, on_progress=None, callback_failure_capacity=crate::interrupt::DEFAULT_CALLBACK_FAILURE_CAPACITY, pulse_extrapolation_ranges=None))]
    fn new(
        seed: u64,
        threads: Option<u32>,
        memory_bytes: Option<u64>,
        cache: &str,
        cache_bytes: Option<u64>,
        kernel: &str,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        callback_failure_capacity: usize,
        pulse_extrapolation_ranges: Option<f64>,
    ) -> PyResult<Self> {
        if let Some(ranges) = pulse_extrapolation_ranges {
            if !(ranges.is_finite() && ranges >= 0.0) {
                return Err(crate::value_err(
                    "pulse_extrapolation_ranges must be finite and nonnegative",
                ));
            }
        }
        if threads == Some(0) {
            return Err(crate::value_err("threads must be positive"));
        }
        let mut context = ExecutionContext::production(seed, crate::resolve_user_threads(threads))
            .with_memory_limit(memory_bytes);
        context.cache_policy = match cache {
            "disabled" => CachePolicy::disabled(),
            "unbounded" => CachePolicy::enabled(None),
            "bounded" => CachePolicy::enabled(Some(
                cache_bytes
                    .ok_or_else(|| crate::value_err("bounded cache requires cache_bytes"))?,
            )),
            _ => return Err(crate::value_err("cache must be disabled, bounded or unbounded")),
        };
        context.kernel_policy = match kernel {
            "auto" => KernelPolicy::production_policy(),
            "scalar" => KernelPolicy::scalar_only(),
            "portable" => KernelPolicy {
                allow_portable_optimized: true,
                allow_arch_simd: false,
                force_scalar: false,
            },
            _ => return Err(crate::value_err("kernel must be auto, scalar or portable")),
        };
        let cancel = cancel.map(|cancel| cancel.inner);
        if let Some(cancel) = &cancel {
            context.cancellation = cancel.clone();
        }
        let progress = crate::callbacks::progress_sink_from_py(on_progress.as_ref())?;
        Ok(Self {
            policy: Arc::new(Policy {
                context,
                progress,
                cancel,
                callback_failure_capacity,
                pulse_extrapolation_ranges,
            }),
        })
    }

    #[getter]
    fn available_memory_bytes(&self) -> u64 {
        self.policy.context.admission().available().bytes()
    }

    /// Invoke one explicitly configured operation. Nested calls restore the
    /// previous profile even when a Python callback or native operation fails.
    #[pyo3(signature = (callable, /, *args, **kwargs))]
    fn call<'py>(
        &self,
        callable: Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _restore = enter(Some(Arc::clone(&self.policy)));
        callable.call(args, kwargs)
    }
}

pub(crate) fn register(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_class::<PythonExecution>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(bytes: u64) -> Arc<Policy> {
        let context = ExecutionContext::for_tests(9).with_memory_limit(Some(bytes));
        Arc::new(Policy {
            context,
            progress: None,
            cancel: None,
            callback_failure_capacity: crate::interrupt::DEFAULT_CALLBACK_FAILURE_CAPACITY,
            pulse_extrapolation_ranges: None,
        })
    }

    #[test]
    fn nested_policies_restore_and_keep_the_same_admission_pool() {
        assert!(capture().is_none());
        let outer = policy(128);
        {
            let _outer = enter(Some(Arc::clone(&outer)));
            let mut first = ExecutionContext::for_tests(21);
            apply(&mut first);
            assert_eq!(first.memory_limit_bytes(), Some(128));
            assert_eq!(first.cache_policy, CachePolicy::disabled());
            let lease = first.acquire_admission(0, 96).unwrap();
            let mut second = ExecutionContext::for_tests(22);
            apply(&mut second);
            assert!(second.acquire_admission_bounded(0, 64, std::time::Duration::ZERO).is_err());
            {
                let _inner = enter(Some(policy(8)));
                let mut inner = ExecutionContext::for_tests(23);
                apply(&mut inner);
                assert_eq!(inner.memory_limit_bytes(), Some(8));
            }
            assert!(Arc::ptr_eq(&capture().unwrap(), &outer));
            drop(lease);
            assert!(second.acquire_admission(0, 64).is_ok());
        }
        assert!(capture().is_none());
    }
}
