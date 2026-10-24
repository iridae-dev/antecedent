//! Measured allocation of the selected-column paths: every ownership copy a design makes
//! is one the receipt or the physical-event record names, and nothing else allocates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "test file: fixtures unwrap, and layout values are compared exactly"
)]

use antecedent_core::VariableId;
use antecedent_core::execution_attempt::{PhysicalEvent, observe_execution};
use antecedent_data::design_layout::{DesignLayout, OwnershipCopy};
use antecedent_data::{ColumnView, TableView, TabularData};

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

const ROWS: usize = 20_000;
/// Heap header and rounding slack for one `Vec` or one `Arc<[f64]>`.
const ALLOCATION_SLACK: u64 = 64;

fn table() -> TabularData {
    let a: Vec<f64> = (0..ROWS).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..ROWS).map(|i| (i * 3) as f64).collect();
    TabularData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice())]).unwrap()
}

fn scattered() -> Vec<bool> {
    (0..ROWS).map(|i| i % 3 != 0).collect()
}

fn bytes(rows: usize) -> u64 {
    (rows * std::mem::size_of::<f64>()) as u64
}

#[test]
fn a_full_mask_and_a_contiguous_run_allocate_nothing() {
    let data = table();
    let a = VariableId::from_raw(0);
    let full = vec![true; ROWS];
    let run: Vec<bool> = (0..ROWS).map(|i| (100..ROWS - 100).contains(&i)).collect();
    for keep in [&full, &run] {
        let (selected, peak) =
            counting::peak_above_start(|| data.float64_selected(a, keep).unwrap());
        assert!(matches!(selected, std::borrow::Cow::Borrowed(_)));
        assert_eq!(peak, 0, "a borrowed selection holds no buffer");
    }
    let attempt = observe_execution(|| data.float64_selected(a, &full));
    attempt.result.unwrap();
    assert_eq!(attempt.report.physical(PhysicalEvent::Copy), None, "no copy is recorded");
}

#[test]
fn a_scattered_mask_makes_exactly_the_copy_it_records() {
    let data = table();
    let a = VariableId::from_raw(0);
    let keep = scattered();
    let kept = keep.iter().filter(|k| **k).count();

    let mut measured = None;
    let attempt = observe_execution(|| {
        measured = Some(counting::peak_above_start(|| data.float64_selected(a, &keep).unwrap()));
        Ok::<(), ()>(())
    });
    attempt.result.unwrap();
    let (selected, peak) = measured.unwrap();
    assert!(matches!(selected, std::borrow::Cow::Owned(_)));
    assert!(peak >= bytes(kept) && peak <= bytes(kept) + ALLOCATION_SLACK, "peak {peak}");
    assert_eq!(attempt.report.physical(PhysicalEvent::Copy), Some(1));
    assert_eq!(attempt.report.physical(PhysicalEvent::CopiedBytes), Some(bytes(kept)));

    // Straight into a design: the destination was allocated by the caller, the gather adds nothing.
    let mut design = vec![0.0; kept * 2];
    let ((), peak) = counting::peak_above_start(|| {
        let (head, tail) = design.split_at_mut(kept);
        data.float64_selected_into(a, &keep, head).unwrap();
        data.float64_selected_into(VariableId::from_raw(1), &keep, tail).unwrap();
    });
    assert_eq!(peak, 0, "gathering into a caller-owned design allocates nothing");
    assert_eq!(&design[..kept], &*data.float64_selected(a, &keep).unwrap());

    // The kept allocation is filled in place: one buffer, not a Vec and then an Arc copy.
    let (shared, peak) =
        counting::peak_above_start(|| data.float64_selected_arc(a, &keep).unwrap());
    assert!(peak >= bytes(kept) && peak <= bytes(kept) + ALLOCATION_SLACK, "peak {peak}");
    assert_eq!(&*shared, &*data.float64_selected(a, &keep).unwrap());
}

#[test]
fn a_masked_layout_lists_every_owned_buffer_and_allocates_no_more() {
    let data = table();
    let views: Vec<ColumnView<'_>> =
        [0, 1].iter().map(|raw| data.column(VariableId::from_raw(*raw)).unwrap()).collect();
    let keep = scattered();
    let kept = keep.iter().filter(|k| **k).count();
    let (layout, peak) =
        counting::peak_above_start(|| DesignLayout::from_mask(&views, &keep).unwrap());
    let copies = layout.receipt().copies();
    assert_eq!(copies.len(), 2);
    assert!(copies.iter().all(|copy| copy.name() == OwnershipCopy::GATHER));
    let named = bytes(kept) * copies.len() as u64;
    assert!(peak >= named, "peak {peak} is below the copies the receipt names");
    assert!(
        peak <= named + 4 * ALLOCATION_SLACK,
        "peak {peak} holds more than the {named} bytes the receipt names"
    );

    let full = vec![true; ROWS];
    let (borrowed, peak) =
        counting::peak_above_start(|| DesignLayout::from_mask(&views, &full).unwrap());
    assert!(borrowed.receipt().is_empty());
    assert!(peak <= 4 * ALLOCATION_SLACK, "a borrowed layout allocates only its column table");
}

#[test]
fn building_a_table_from_slices_copies_each_column_once() {
    let columns: Vec<Vec<f64>> =
        (0..3).map(|c| (0..ROWS).map(|i| (i + c) as f64).collect()).collect();
    let (data, peak) = counting::peak_above_start(|| {
        TabularData::from_f64_columns(
            ["a", "b", "c"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap()
    });
    let owned = bytes(ROWS) * 3;
    assert_eq!(data.row_count(), ROWS);
    assert!(peak >= owned, "peak {peak} is below the three owned columns");
    assert!(
        peak <= owned + 64 * 1024,
        "peak {peak} holds a temporary copy beyond the {owned} owned bytes"
    );
}
