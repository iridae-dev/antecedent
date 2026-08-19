//! Allocation-counting harness for the gather hot path.
//!
//! Scoped to this integration-test binary so the rest of the workspace is not
//! forced onto a `#[global_allocator]`. ADR 0011's allocation assertions for
//! gather are this count, not only output-buffer pointer stability.
//!
//! The counter is **per-thread**: gather is single-threaded (nothing in
//! `antecedent-kernels` spawns), so counting on the calling thread keeps the
//! assertion at full strength while making it immune to background
//! allocations from the test harness's own threads — a global counter
//! intermittently picked those up in CI and failed the run.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use antecedent_core::KernelPolicy;
use antecedent_kernels::{F64VectorView, gather};

struct CountingAlloc;

thread_local! {
    // const-initialized: no lazy allocation on first access, so counting
    // inside the allocator cannot recurse.
    static THREAD_ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn count_one() {
    // try_with: allocations during TLS teardown must not panic.
    let _ = THREAD_ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_one();
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_one();
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_one();
        // SAFETY: `ptr` came from this allocator; layout matches the original allocation.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator; layout matches the original allocation.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[test]
fn gather_into_presized_buffer_allocates_nothing() {
    let n = 8_000usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let src = F64VectorView::contiguous(&data);
    let indices: Vec<usize> = (0..n).step_by(8).collect();
    let mut out = vec![0.0; indices.len()];
    let policy = KernelPolicy::default_policy();
    gather(&policy, src, &indices, &mut out);

    let before = THREAD_ALLOCATIONS.with(Cell::get);
    for _ in 0..200 {
        gather(&policy, src, &indices, &mut out);
    }
    let after = THREAD_ALLOCATIONS.with(Cell::get);
    assert_eq!(
        after, before,
        "gather into a pre-sized buffer must not heap-allocate (before={before} after={after})"
    );
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 8.0);
}
