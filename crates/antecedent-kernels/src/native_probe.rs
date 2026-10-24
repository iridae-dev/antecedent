//! Process-level resource probe for native workloads: a counting global allocator and
//! peak resident set size.
//!
//! `antecedent-core` forbids `unsafe_code`, so the two things that need it live here and
//! plug into `antecedent_core::measurement_record::measure` through [`NativeProbe`]:
//!
//! - [`CountingAllocator`] wraps the system allocator and counts every byte and call the
//!   whole process makes, across all threads. A binary opts in with
//!   `#[global_allocator] static ALLOC: CountingAllocator = CountingAllocator::new();`.
//!   Libraries never install it. Without it [`NativeProbe`] reports no allocation figures
//!   and the record lists them as unobserved.
//! - [`peak_rss_bytes`] reads the process high-water resident set with `getrusage`
//!   (`ru_maxrss`: bytes on macOS, kibibytes elsewhere), declared directly so no `libc`
//!   dependency is added. Hosts without it report an explicit reason.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use antecedent_core::measurement_record::{AllocationSample, PeakRss, ResourceProbe};

static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// `getrusage` is not declared for this target (needs a 64-bit Unix host).
pub const PEAK_RSS_NO_GETRUSAGE: &str =
    "peak RSS is unsupported: getrusage(RUSAGE_SELF) is only read on 64-bit Unix hosts";

/// `getrusage` returned an error.
pub const PEAK_RSS_GETRUSAGE_FAILED: &str = "peak RSS is unsupported: getrusage failed";

fn note_allocation(bytes: usize) {
    let bytes = bytes as u64;
    TOTAL_BYTES.fetch_add(bytes, Ordering::Relaxed);
    CALLS.fetch_add(1, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed).saturating_add(bytes);
    PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn note_release(bytes: usize) {
    let bytes = bytes as u64;
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
        Some(live.saturating_sub(bytes))
    });
}

/// The system allocator plus process-wide byte and call counters.
#[derive(Debug, Default)]
pub struct CountingAllocator;

impl CountingAllocator {
    /// A counting allocator, usable in a `static`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: every method forwards to `System` with the arguments it was given; the counters
// are atomics and never touch the allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same layout as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_allocation(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same layout as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_allocation(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from this allocator with `layout`; `new_size` is the caller's.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            note_release(layout.size());
            note_allocation(new_size);
        }
        moved
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        note_release(layout.size());
    }
}

/// Allocator totals since the process started, or `None` when [`CountingAllocator`] is not
/// the global allocator (no call has been counted).
#[must_use]
pub fn allocation_sample() -> Option<AllocationSample> {
    let count = CALLS.load(Ordering::Relaxed);
    (count > 0).then(|| AllocationSample {
        total_bytes: TOTAL_BYTES.load(Ordering::Relaxed),
        count,
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
    })
}

#[cfg(all(unix, target_pointer_width = "64"))]
unsafe extern "C" {
    // `usage` is a `struct rusage`: two 16-byte timevals, then `ru_maxrss` as a 64-bit long.
    fn getrusage(who: i32, usage: *mut i64) -> i32;
}

/// Peak resident set size of this process so far, from `getrusage(RUSAGE_SELF)`.
///
/// The figure is the process lifetime high-water mark. Read it before and after a run to
/// tell whether the run raised it, or run each repeat in its own process.
#[must_use]
pub fn peak_rss_bytes() -> PeakRss {
    #[cfg(all(unix, target_pointer_width = "64"))]
    {
        // 256 bytes is larger than `struct rusage` on every supported Unix.
        let mut usage = [0i64; 32];
        // SAFETY: `usage` is writable and larger than `struct rusage`; RUSAGE_SELF is 0.
        let status = unsafe { getrusage(0, usage.as_mut_ptr()) };
        if status != 0 {
            return PeakRss::Unsupported { reason: PEAK_RSS_GETRUSAGE_FAILED };
        }
        // ru_utime and ru_stime take the first four 8-byte slots; ru_maxrss follows.
        let raw = u64::try_from(usage[4]).unwrap_or(0);
        if cfg!(target_os = "macos") {
            PeakRss::Bytes(raw)
        } else {
            PeakRss::Bytes(raw.saturating_mul(1024))
        }
    }
    #[cfg(not(all(unix, target_pointer_width = "64")))]
    {
        PeakRss::Unsupported { reason: PEAK_RSS_NO_GETRUSAGE }
    }
}

/// The probe for native workloads: [`CountingAllocator`] totals and [`peak_rss_bytes`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeProbe;

impl ResourceProbe for NativeProbe {
    fn begin_window(&self) {
        PEAK_LIVE_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    fn allocation(&self) -> Option<AllocationSample> {
        allocation_sample()
    }

    fn peak_rss(&self) -> PeakRss {
        peak_rss_bytes()
    }
}

/// Write `json` to `<root>/<bench>/measurement.json`, the file `scripts/bench_regression.py`
/// attaches to the Criterion comparison of that bench. Returns the path written.
///
/// # Errors
///
/// The directory cannot be created or the file cannot be written.
pub fn write_measurement(root: &Path, bench: &str, json: &str) -> std::io::Result<PathBuf> {
    let dir = root.join(bench);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("measurement.json");
    std::fs::write(&path, json)?;
    Ok(path)
}
