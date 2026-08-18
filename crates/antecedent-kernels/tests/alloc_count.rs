//! Allocation-counting harness for the gather hot path.
//!
//! Scoped to this integration-test binary so the rest of the workspace is not
//! forced onto a `#[global_allocator]`. ADR 0011's allocation assertions for
//! gather are this count, not only output-buffer pointer stability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use antecedent_core::KernelPolicy;
use antecedent_kernels::{F64VectorView, gather};

struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
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

    let before = ALLOCATIONS.load(Ordering::SeqCst);
    for _ in 0..200 {
        gather(&policy, src, &indices, &mut out);
    }
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    assert_eq!(
        after, before,
        "gather into a pre-sized buffer must not heap-allocate (before={before} after={after})"
    );
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 8.0);
}
