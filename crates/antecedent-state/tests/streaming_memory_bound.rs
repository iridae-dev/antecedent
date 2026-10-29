//! The streaming summaries are bounded end to end: the bytes an append holds above its
//! start do not grow with the number of rows streamed through it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "test file: fixtures unwrap, and the streamed result is compared exactly"
)]

use antecedent_state::{LinearOlsSuffStats, StreamingCovariance};

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

const COLUMNS: usize = 4;

fn row(i: usize) -> [f64; COLUMNS] {
    let x = i as f64;
    [1.0, (x * 0.37).sin(), (x * 0.11).cos(), (x % 7.0) - 3.0]
}

fn response(i: usize) -> f64 {
    1.0 + 2.0 * row(i)[1] - 0.5 * row(i)[3] + ((i * 13 % 17) as f64) / 17.0
}

/// Peak bytes held above the start while `rows` rows stream through a fresh covariance.
fn covariance_peak(rows: usize) -> (u64, StreamingCovariance) {
    let mut state = StreamingCovariance::new(COLUMNS).unwrap();
    let ((), peak) = counting::peak_above_start(|| {
        for i in 0..rows {
            state.append(&row(i)).unwrap();
        }
    });
    (peak, state)
}

fn ols_peak(rows: usize) -> (u64, LinearOlsSuffStats) {
    let mut state = LinearOlsSuffStats::new(COLUMNS).unwrap();
    let ((), peak) = counting::peak_above_start(|| {
        for i in 0..rows {
            state.append_row(&row(i), response(i)).unwrap();
        }
    });
    (peak, state)
}

#[test]
fn streaming_covariance_holds_nothing_that_grows_with_the_rows() {
    let (small, _) = covariance_peak(1_000);
    let (large, state) = covariance_peak(200_000);
    assert_eq!(small, 0, "an append updates the mean and scatter in place");
    assert_eq!(large, 0, "200000 rows hold no more than 1000 rows");
    assert_eq!(state.n(), 200_000);
    assert_eq!(state.m2().len(), COLUMNS * COLUMNS);
}

#[test]
fn streaming_ols_statistics_are_bounded_by_the_design_width_not_the_rows() {
    let (small, _) = ols_peak(1_000);
    let (large, state) = ols_peak(100_000);
    // Each append stages one joint row; nothing is retained per row.
    let staged = ((COLUMNS + 1) * std::mem::size_of::<f64>()) as u64;
    assert!(small >= staged && small <= 4 * staged, "peak {small}");
    assert_eq!(large, small, "100000 rows hold no more than 1000 rows");
    assert_eq!(state.n(), 100_000);
    // The streamed summary is the batch solve of the same rows.
    let beta = state.solve_beta().unwrap();
    let mut batch = LinearOlsSuffStats::new(COLUMNS).unwrap();
    let rows: Vec<f64> = (0..100_000).flat_map(row).collect();
    let y: Vec<f64> = (0..100_000).map(response).collect();
    batch.append_batch(&rows, &y).unwrap();
    assert_eq!(beta, batch.solve_beta().unwrap());
    // numpy.linalg.lstsq of the same 100000 rows (an independent solve), to 1e-9.
    let reference = [
        1.470_584_831_508_133_3,
        1.996_360_620_158_106,
        -4.126_180_007_116_41e-6,
        -0.499_996_014_824_548_53,
    ];
    for (got, want) in beta.iter().zip(reference) {
        assert!((got - want).abs() < 1e-9, "{beta:?} vs {reference:?}");
    }
}
