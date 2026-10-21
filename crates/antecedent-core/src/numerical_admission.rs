// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carry a public workflow's admission authority through context-free numerical traits.
//! This is scoped thread state, never shared mutable prepared state or a second budget.
use crate::{
    ExecutionContext,
    execution::{AdmissionError, AdmissionLease},
};
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Default)]
struct Binding {
    context: Option<ExecutionContext>,
    reservation: Option<Arc<AdmissionLease>>,
    worker: Option<Arc<AdmissionLease>>,
}
thread_local! { static ACTIVE: RefCell<Binding> = RefCell::default(); }
struct Restore(Binding);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = std::mem::take(&mut self.0));
    }
}
/// Worker-carried authority for the same pool and any already admitted workspace.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct NumericalAdmissionToken(Binding);
impl NumericalAdmissionToken {
    /// Capture the current workflow binding.
    #[must_use]
    pub fn capture() -> Self {
        Self(ACTIVE.with(|active| active.borrow().clone()))
    }
    /// Install this binding on a worker, using its bounded inner context.
    pub fn execute<T>(&self, ctx: &ExecutionContext, execute: impl FnOnce() -> T) -> T {
        let binding = Binding {
            context: Some(ctx.clone()),
            reservation: self.0.reservation.clone(),
            worker: self.0.worker.clone(),
        };
        let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
        execute()
    }
}
/// Install the caller's execution policy while preserving an enclosing workspace reservation.
#[doc(hidden)]
pub fn with_context<T>(ctx: &ExecutionContext, execute: impl FnOnce() -> T) -> T {
    NumericalAdmissionToken::capture().execute(ctx, execute)
}
/// Make the already admitted compiled workspace available to original numerical kernels.
#[doc(hidden)]
pub fn with_reservation<T>(reservation: Arc<AdmissionLease>, execute: impl FnOnce() -> T) -> T {
    let mut binding = ACTIVE.with(|active| active.borrow().clone());
    binding.reservation = Some(reservation);
    let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
    execute()
}
/// Draw actual scratch bytes from the current workflow. Context-free component calls return None.
#[doc(hidden)]
#[must_use]
pub fn acquire(bytes: u64) -> Option<Result<AdmissionLease, AdmissionError>> {
    let binding = ACTIVE.with(|active| active.borrow().clone());
    if let Some(reservation) = binding.reservation {
        Some(reservation.acquire_child(0, bytes))
    } else {
        binding.context.map(|context| context.acquire_admission(0, bytes))
    }
}

/// Check the active workflow's cancellation at an original numerical loop boundary.
#[doc(hidden)]
#[must_use]
pub fn is_cancelled() -> bool {
    ACTIVE.with(|active| {
        active.borrow().context.as_ref().is_some_and(|ctx| ctx.cancellation.is_cancelled())
    })
}

/// Current worker reservation, retained only for the dynamic worker scope.
pub(crate) fn worker() -> Option<Arc<AdmissionLease>> {
    ACTIVE.with(|active| active.borrow().worker.clone())
}
/// Install the real worker lease so nested admitted maps reuse that CPU slot.
pub(crate) fn with_worker<T>(worker: Arc<AdmissionLease>, execute: impl FnOnce() -> T) -> T {
    let mut binding = ACTIVE.with(|active| active.borrow().clone());
    binding.worker = Some(worker);
    let _restore = Restore(ACTIVE.with(|active| active.replace(binding)));
    execute()
}
