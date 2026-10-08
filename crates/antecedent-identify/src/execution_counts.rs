//! Instrumentation of actual finite generic-ID and multi-source checker invocations.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::Cell;
thread_local! {static CHECKS:Cell<u64>=const{Cell::new(0)};}
/// Observe actual primitive checker invocations, including nested scopes and refusals.
pub fn count_checked_identifications<R>(work: impl FnOnce() -> R) -> (R, u64) {
    let before = CHECKS.with(Cell::get);
    let result = work();
    (result, CHECKS.with(Cell::get).saturating_sub(before))
}
pub(crate) fn note_check() {
    CHECKS.with(|count| count.set(count.get().saturating_add(1)));
}
