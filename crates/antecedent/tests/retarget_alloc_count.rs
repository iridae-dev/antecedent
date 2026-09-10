//! Allocation-counting harness for the retarget hot path.
//!
//! Scoped to this integration-test binary so the rest of the workspace is not
//! forced onto a `#[global_allocator]`. ADR 0011's allocation assertions for
//! retarget are this count, not only the timed 10k×500 pin.
//!
//! Counts `antecedent_estimate::retarget` after prepare / warmup. Result
//! assembly lives outside the numerical hot path and is not included.
//!
//! The counter is **per-thread**: retarget is single-threaded, so counting on
//! the calling thread keeps the assertion at full strength while making it
//! immune to background allocations from the test harness.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]
#![allow(clippy::many_single_char_names)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::retarget;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

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

/// Per-call scratch for summarize / joint IF / contrast / descendant check.
/// Measured 53 on this 10k×2-column table; 64 leaves platform slack without
/// hiding an O(n) container-per-row regression.
const MAX_ALLOCS_PER_RETARGET: u64 = 64;
const N_ROWS: usize = 10_000;
const N_LEVERS: u64 = 500;

#[test]
fn retarget_10k_by_500_stays_within_allocation_budget() {
    let n = N_ROWS;
    let mut rng = ExecutionContext::for_tests(26).rng.stream(0x15);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p = 1.0 / (1.0 + (-(-0.3 + 0.9 * zi)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + 0.4 * zi + 0.35 * standard_normal(&mut rng);
    }
    let mut levers = Vec::with_capacity(N_LEVERS as usize);
    for k in 0..N_LEVERS {
        let center = -2.0 + 4.0 * k as f64 / (N_LEVERS as f64 - 1.0);
        let mut w = vec![0.0; n];
        for i in 0..n {
            w[i] = (-0.5 * ((z[i] - center) / 0.7).powi(2)).exp();
        }
        levers.push(w);
    }
    let borrowed = [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())];
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(26);
    let prepared = Study::tabular(data)
        .graph(graph.clone())
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let table = prepared.score_table().expect("AIPW prepare must freeze a score table");
    let z_id = VariableId::from_raw(2);
    let depends = [z_id];
    let (warmup, _) = retarget(table, &levers[0], &depends, Some(&graph), Some(&t), None).unwrap();
    assert!(warmup.summary.means.iter().all(|m| m.is_finite()));

    let before = THREAD_ALLOCATIONS.with(Cell::get);
    for w in &levers {
        let (out, _) = retarget(table, w, &depends, Some(&graph), Some(&t), None).unwrap();
        assert!(out.summary.means.iter().all(|m| m.is_finite()));
    }
    let after = THREAD_ALLOCATIONS.with(Cell::get);
    let used = after.saturating_sub(before);
    let budget = MAX_ALLOCS_PER_RETARGET * N_LEVERS;
    assert!(
        used <= budget,
        "retarget 10k×500 allocated {used} times (budget {budget}; \
         {MAX_ALLOCS_PER_RETARGET} per call)"
    );
}
