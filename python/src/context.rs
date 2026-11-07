//! Python execution-context setup and cooperative cancellation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CachePolicy, ExecutionContext, execution_api, interrupt, pyclass, pymethods};

/// Default coalition / semantic cache budget for Python production contexts
/// (matches attribution bench policy).
pub(crate) const PY_DEFAULT_CACHE_MAX_BYTES: u64 = 4_000_000;

/// Resolve an omitted or zero `threads=` to [`antecedent::execution::default_user_threads`].
/// An explicit `threads >= 1` is a pin and may exceed the default cap.
pub(crate) fn resolve_user_threads(threads: Option<u32>) -> u32 {
    match threads {
        Some(n) if n >= 1 => n,
        Some(_) => 1,
        None => antecedent::execution::default_user_threads(),
    }
}

pub(crate) fn py_execution_context(seed: u64, threads: u32) -> ExecutionContext {
    py_execution_context_ext(seed, threads, None, None, Some(PY_DEFAULT_CACHE_MAX_BYTES))
}

/// [`py_execution_context`] honouring an optional caller token.
pub(crate) fn py_execution_context_cancel(
    seed: u64,
    threads: u32,
    cancel: Option<PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = py_execution_context(seed, threads);
    apply_cancel(&mut ctx, cancel);
    ctx
}

pub(crate) fn py_execution_context_ext(
    seed: u64,
    threads: u32,
    cancel: Option<antecedent::execution::CancellationToken>,
    progress: Option<std::sync::Arc<dyn antecedent::execution::ProgressSink>>,
    cache_max_bytes: Option<u64>,
) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(seed, threads);
    ctx.cache_policy = CachePolicy::enabled(cache_max_bytes);
    install_cancellation(&mut ctx, cancel);
    ctx.progress = interrupt::bind_progress(progress);
    execution_api::apply(&mut ctx);
    ctx
}

/// Install the caller's cancellation token on `ctx`; Ctrl-C also fires it (or, with no token, the
/// run's own).
pub(crate) fn apply_cancel(ctx: &mut ExecutionContext, cancel: Option<PyCancellationToken>) {
    install_cancellation(ctx, cancel.map(|token| token.inner));
}

pub(crate) fn install_cancellation(
    ctx: &mut ExecutionContext,
    cancel: Option<antecedent::execution::CancellationToken>,
) {
    if let Some(token) = cancel {
        interrupt::link(&token);
        ctx.cancellation = token;
    } else if let Some(token) = interrupt::ambient_token() {
        ctx.cancellation = token;
    } else {
        // Built ahead of the detached call: the token joins the run that executes this context.
        interrupt::link(&ctx.cancellation);
    }
}

/// Cooperative cancellation token shared with a running analysis.
#[pyclass(name = "CancellationToken", from_py_object)]
#[derive(Clone)]
pub struct PyCancellationToken {
    pub(crate) inner: antecedent::execution::CancellationToken,
}

#[pymethods]
impl PyCancellationToken {
    #[new]
    fn new() -> Self {
        Self { inner: antecedent::execution::CancellationToken::new() }
    }

    fn cancel(&self) {
        self.inner.cancel();
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}
