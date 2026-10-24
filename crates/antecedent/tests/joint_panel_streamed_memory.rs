//! The tiled panel reduction is bounded end to end: with one draw per tile it holds less
//! than the full-output route that keeps every draw by world by period, and its receipt
//! never reports more live draws than the tile.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test file: fixtures unwrap")]

#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use antecedent::Study;
use antecedent::analysis::joint_panel::JointPanelResponseFunctional;
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_panel::PanelInference;

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

#[test]
fn tiled_projection_holds_less_than_the_full_output_route() {
    let mut ctx = ExecutionContext::for_tests(20_261_101);
    ctx.cache_policy = antecedent_core::execution::CachePolicy::enabled(None);
    let data = fixture::source(false);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 100 });
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let functional = JointPanelResponseFunctional::WindowChange { before: (0, 5), after: (18, 23) };
    let worlds = result.request().worlds.clone();
    let target = result.request().target.clone();

    let (streamed, streamed_peak) =
        counting::peak_above_start(|| result.project_streamed(&functional, 1, &ctx).unwrap());
    let (retained, retained_peak) =
        counting::peak_above_start(|| result.retarget_frozen(worlds, target, &ctx).unwrap());
    assert!(streamed.receipt().peak_tile_draws <= 1);
    assert_eq!(streamed.draws().len(), retained.draws.len());
    assert!(streamed_peak > 0 && retained_peak > 0);
    assert!(
        streamed_peak < retained_peak,
        "tiled peak {streamed_peak} is not below the full-output peak {retained_peak}"
    );
}
