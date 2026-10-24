//! Outcomes-only counterfactual streaming is bounded end to end: it returns the retained
//! route's outcome column bit for bit while holding fewer bytes than the route that keeps
//! every node column.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test file: fixtures unwrap")]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Intervention, Value, VariableId};
use antecedent_counterfactual::{
    CounterfactualEngine, CounterfactualWorld, ExogenousPosterior, NoiseInferenceKind,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, MechanismWorkspace, ModelView,
    NoiseTable,
};

/// Per-thread live and peak byte counts, so concurrent tests do not see each
/// other. Every call forwards to the system allocator unchanged.
#[allow(unsafe_code, reason = "test-only counting allocator that forwards to the system allocator")]
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct Counting;

    thread_local! {
        // const-initialized: no lazy allocation, so counting cannot recurse.
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    fn grow(bytes: usize) {
        let delta = isize::try_from(bytes).unwrap_or(isize::MAX);
        let _ = LIVE.try_with(|live| {
            let now = live.get().saturating_add(delta);
            live.set(now);
            let _ = PEAK.try_with(|peak| {
                if now > peak.get() {
                    peak.set(now);
                }
            });
        });
    }

    fn shrink(bytes: usize) {
        let delta = isize::try_from(bytes).unwrap_or(isize::MAX);
        let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(delta)));
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwarding to the system allocator with the same layout.
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                grow(layout.size());
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwarding to the system allocator with the same layout.
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                grow(layout.size());
            }
            ptr
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: `ptr` came from this allocator with this layout.
            let moved = unsafe { System.realloc(ptr, layout, new_size) };
            if !moved.is_null() {
                shrink(layout.size());
                grow(new_size);
            }
            moved
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: `ptr` came from this allocator with this layout.
            unsafe { System.dealloc(ptr, layout) };
            shrink(layout.size());
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    /// Run `work` and report the most bytes this thread held above where it started.
    pub fn peak_above_start<T>(work: impl FnOnce() -> T) -> (T, u64) {
        let start = LIVE.with(Cell::get);
        PEAK.with(|peak| peak.set(start));
        let out = work();
        let peak = PEAK.with(Cell::get);
        (out, u64::try_from(peak.saturating_sub(start)).unwrap_or(0))
    }
}

const ROWS: usize = 200_000;

fn chain() -> (CounterfactualEngine, ExogenousPosterior) {
    let mut g = Dag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let linear = |intercept: f64, coeffs: &[f64], sigma: f64| MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(coeffs.to_vec()),
        sigma,
    };
    let model = CompiledCausalModel::compile(g).unwrap().with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            linear(1.0, &[], 1.0),
            linear(0.5, &[2.0], 0.5),
            linear(-1.0, &[3.0], 2.0),
        ]),
    });
    let ctx = ExecutionContext::for_tests(3_076_196);
    let table = NoiseTable::sample(&ModelView::observational(&model), ROWS, &ctx).unwrap();
    let exo = ExogenousPosterior {
        noise: Arc::from(table.values()),
        n_units: ROWS,
        n_nodes: 3,
        kind: NoiseInferenceKind::Invertible,
        assumed_columns: Arc::from([false, false, false]),
    };
    (CounterfactualEngine::new(model), exo)
}

#[test]
fn outcomes_only_streaming_holds_less_than_retaining_every_node() {
    let (engine, exo) = chain();
    let ctx = ExecutionContext::for_tests(1);
    let outcome = VariableId::from_raw(2);
    let world = CounterfactualWorld {
        unit_rows: None,
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    };
    let worlds = std::slice::from_ref(&world);
    let mut workspace = MechanismWorkspace::default();
    // A first run sizes the reusable workspace, so both measurements start from the same state.
    engine.predict(&exo, worlds, &[outcome], false, &mut workspace, &ctx).unwrap();

    let (retained, retained_peak) = counting::peak_above_start(|| {
        engine.predict(&exo, worlds, &[outcome], false, &mut workspace, &ctx).unwrap()
    });
    let (streamed, streamed_peak) = counting::peak_above_start(|| {
        engine
            .predict_retaining_outcomes(&exo, worlds, &[outcome], false, &mut workspace, &ctx)
            .unwrap()
    });
    let node = DenseNodeId::from_raw(2);
    let kept: Vec<u64> =
        retained.outcome_column(0, node).unwrap().iter().map(|v| v.to_bits()).collect();
    let streamed_bits: Vec<u64> =
        streamed.outcome_column(0, node).unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(kept, streamed_bits, "streaming must equal retained");
    assert!(streamed_peak > 0);
    let column = (ROWS * std::mem::size_of::<f64>()) as u64;
    assert!(
        streamed_peak + column <= retained_peak,
        "outcomes-only peak {streamed_peak} is not at least one node column below {retained_peak}"
    );
}
