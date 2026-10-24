//! Native measurement of real work: allocator bytes and calls, live-byte peak and peak
//! resident set size around code that actually runs.
//!
//! The counting allocator is process-wide, so this binary holds a single test that runs its
//! windows one after another. Assertions on bytes are lower bounds from the work built in
//! the window, or small ceilings for work that must not allocate, so harness threads cannot
//! make them flaky.
//!
//! Set `ANTECEDENT_MEASUREMENT_OUT` to a directory to also write the summary of the
//! scaled windows as `<dir>/native_probe_scaling/measurement.json`; the Python schema test
//! reads that file.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "test scaffolding sizes buffers from small literals and compares exact constants"
)]

use std::hint::black_box;
use std::path::PathBuf;

use antecedent_core::ExecutionContext;
use antecedent_core::KernelPolicy;
use antecedent_core::measurement_record::{
    MeasurementRecord, MeasurementSummary, PeakRss, PeakRssSummary, measure,
};
use antecedent_kernels::native_probe::{
    CountingAllocator, NativeProbe, allocation_sample, peak_rss_bytes, write_measurement,
};
use antecedent_kernels::{F64VectorView, gather};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator::new();

const MIB: u64 = 1 << 20;

fn context() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

/// Build and fill an owned buffer of `words` u64 values, then drop it.
fn build_buffer(words: usize) -> Result<usize, ()> {
    let buffer = vec![7u64; words];
    Ok(black_box(&buffer).len())
}

fn window(words: usize) -> MeasurementRecord {
    let (result, record) = measure(&context(), &NativeProbe, || build_buffer(words));
    assert_eq!(result, Ok(words));
    record
}

#[test]
fn the_probe_reads_the_allocator_and_the_resident_set_around_real_work() {
    assert!(allocation_sample().is_some(), "the counting allocator is the global allocator");

    // Building a buffer shows up as at least the buffer's bytes, a live peak of at least
    // that size, and at least one allocation call.
    let words = (4 * MIB / 8) as usize;
    let built = window(words);
    assert!(built.observed("allocation_bytes"));
    assert!(built.allocation_bytes >= 4 * MIB, "allocated {}", built.allocation_bytes);
    assert!(built.peak_live_bytes >= 4 * MIB, "live peak {}", built.peak_live_bytes);
    assert!(built.allocation_count >= 1);

    // The bytes follow the work: a buffer four times as large allocates at least three
    // buffers' worth more.
    let larger = window(4 * words);
    assert!(
        larger.allocation_bytes >= 4 * 4 * MIB,
        "{} vs {}",
        larger.allocation_bytes,
        built.allocation_bytes
    );
    assert!(larger.peak_live_bytes >= 16 * MIB);

    // Gather into a pre-sized output buffer must not allocate; the window sees only the
    // observer's own bookkeeping.
    let n = 8_000usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let src = F64VectorView::contiguous(&data);
    let indices: Vec<usize> = (0..n).step_by(8).collect();
    let mut out = vec![0.0; indices.len()];
    let policy = KernelPolicy::default_policy();
    let (result, gathered) = measure(&context(), &NativeProbe, || {
        for _ in 0..200 {
            gather(&policy, src, &indices, &mut out)?;
        }
        Ok::<_, antecedent_kernels::ViewError>(())
    });
    result.unwrap();
    assert_eq!(out[1], 8.0);
    assert!(
        gathered.allocation_bytes < 64 * 1024,
        "pre-sized gather allocated {} bytes",
        gathered.allocation_bytes
    );

    // Resident set: touching 48 MiB page by page raises the process high-water mark to at
    // least that much. The mark never goes down.
    let before = peak_rss_bytes();
    let (touched, resident) = measure(&context(), &NativeProbe, || {
        let mut pages = vec![0u8; (48 * MIB) as usize];
        for at in (0..pages.len()).step_by(4096) {
            pages[at] = 1;
        }
        Ok::<_, ()>(black_box(&pages).iter().step_by(4096).map(|&b| usize::from(b)).sum::<usize>())
    });
    assert_eq!(touched, Ok((48 * MIB / 4096) as usize));
    match (before, resident.peak_rss) {
        (PeakRss::Bytes(then), PeakRss::Bytes(now)) => {
            assert!(now >= then, "peak resident set fell from {then} to {now}");
            assert!(now >= 48 * MIB, "peak resident set {now} is below the pages touched");
        }
        (_, PeakRss::Unsupported { reason }) => {
            assert!(!cfg!(all(unix, target_pointer_width = "64")), "{reason}");
            assert!(!reason.is_empty());
        }
        _ => panic!("peak resident set sampled before the run but not after"),
    }

    // Repeats at three sizes reduce to the spread of the allocation they performed.
    let repeats: Vec<MeasurementRecord> =
        [MIB / 8, 2 * MIB / 8, 8 * MIB / 8].iter().map(|&words| window(words as usize)).collect();
    let summary = MeasurementSummary::from_repeats(&repeats);
    assert_eq!(summary.allocation_bytes.count, 3);
    assert!(summary.allocation_bytes.min >= MIB);
    assert!(summary.allocation_bytes.max >= 8 * MIB);
    assert!(summary.allocation_bytes.max > summary.allocation_bytes.min);
    if cfg!(all(unix, target_pointer_width = "64")) {
        assert!(matches!(summary.peak_rss, PeakRssSummary::Bytes(_)));
    }

    if let Some(root) = std::env::var_os("ANTECEDENT_MEASUREMENT_OUT") {
        let path =
            write_measurement(&PathBuf::from(root), "native_probe_scaling", &summary.to_json())
                .unwrap();
        assert!(path.is_file());
    }
}
