//! Explicit counters for one native workload repeat.
//!
//! Instrumented code reports these figures. Nothing here hooks the allocator,
//! and nothing infers them from a Python trace. [`MeasurementSummary`] reduces
//! repeats to min / median / max, matching the median summaries in the bench
//! reports. Even-count medians use [`u64::midpoint`] (floor of the two central
//! values).
//!
//! Peak resident set size on macOS is `task_info` with `TASK_VM_INFO`
//! (`resident_size_peak`) or `getrusage(RUSAGE_SELF).ru_maxrss` (bytes). Both
//! need `libc` and `unsafe`. This crate forbids `unsafe_code` and does not
//! depend on `libc`, so non-Linux hosts record [`PeakRss::Unsupported`]. Linux
//! reads `VmHWM` (high-water resident set, KiB) from `/proc/self/status` with
//! `std::fs` only.
//!
//! Field names match the optional `measurement` object
//! `scripts/bench_regression.py` attaches when `<bench>/measurement.json` is
//! present. This module is not a second benchmark runner.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Call sites arrive with later physical-plan work. `cargo test` builds the
// library without `cfg(test)`, and CI sets `-D warnings`.
#![allow(dead_code)]

/// Not sampled until [`MeasurementRecord::observe_peak_rss`] or an explicit set.
pub(crate) const PEAK_RSS_NOT_SAMPLED: &str =
    "peak RSS was not sampled; report it explicitly or call observe_peak_rss";

/// macOS `task_info` / `getrusage` are unavailable without `libc`.
pub(crate) const PEAK_RSS_NEEDS_LIBC: &str = concat!(
    "peak RSS is unsupported here: macOS reads it with mach task_info ",
    "(TASK_VM_INFO resident_size_peak) or getrusage(RUSAGE_SELF).ru_maxrss, ",
    "and Linux getrusage is the libc syscall; antecedent-core forbids unsafe_code ",
    "and does not depend on libc. Linux VmHWM from /proc/self/status is the std-only path"
);

/// `/proc/self/status` did not contain a `VmHWM` kibibyte field.
#[cfg(target_os = "linux")]
pub(crate) const PEAK_RSS_VMHWM_MISSING: &str =
    "peak RSS is unsupported: /proc/self/status has no VmHWM kibibyte field";

/// `/proc/self/status` could not be read.
#[cfg(target_os = "linux")]
pub(crate) const PEAK_RSS_PROC_UNREADABLE: &str =
    "peak RSS is unsupported: /proc/self/status could not be read";

/// Repeats disagreed, or only some of them had a sample.
pub(crate) const PEAK_RSS_INCOMPLETE: &str = "peak RSS was not available for every repeat";

/// [`MeasurementSummary::from_repeats`] on an empty slice.
pub(crate) const PEAK_RSS_NO_REPEATS: &str = "no measurement repeats";

/// Peak resident set, or why this process cannot read it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PeakRss {
    /// High-water resident set size in bytes.
    Bytes(u64),
    /// No sample. `reason` is a stable explanation, not a placeholder.
    Unsupported {
        /// Why peak RSS was not read.
        reason: &'static str,
    },
}

impl Default for PeakRss {
    fn default() -> Self {
        Self::Unsupported { reason: PEAK_RSS_NOT_SAMPLED }
    }
}

/// One named stage's wall time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StageTime {
    /// Stage id, for example `estimate`.
    pub name: String,
    /// Nanoseconds reported for this stage in the repeat.
    pub nanos: u64,
}

/// Counters one native repeat reports. Zero means none reported, not "unknown".
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MeasurementRecord {
    /// Peak RSS sample for this repeat.
    pub peak_rss: PeakRss,
    /// Bytes the instrumented native code allocated.
    pub allocation_bytes: u64,
    /// Bytes copied into owned native buffers.
    pub copy_bytes: u64,
    /// Bytes borrowed, not copied.
    pub borrow_bytes: u64,
    /// Passes over the source.
    pub source_passes: u64,
    /// Model fits.
    pub fit_count: u64,
    /// Tests or checks.
    pub test_count: u64,
    /// Outputs kept after the repeat.
    pub retained_outputs: u64,
    /// Bytes spilled to host or disk.
    pub spill_bytes: u64,
    /// Worker threads used. Zero means the repeat did not report a count.
    pub threads: u32,
    /// Device name. `None` is the host CPU.
    pub device: Option<String>,
    /// Per-stage wall times. The same name within a repeat is summed.
    pub stage_times_ns: Vec<StageTime>,
    /// Cooperative cancellation was observed.
    pub cancelled: bool,
    /// The repeat failed.
    pub failed: bool,
}

impl MeasurementRecord {
    /// Add bytes of a native allocation the caller actually performed.
    pub(crate) fn add_allocation_bytes(&mut self, bytes: u64) {
        self.allocation_bytes = self.allocation_bytes.saturating_add(bytes);
    }

    /// Add bytes of a native copy the caller actually performed.
    pub(crate) fn add_copy_bytes(&mut self, bytes: u64) {
        self.copy_bytes = self.copy_bytes.saturating_add(bytes);
    }

    /// Add bytes of a native borrow the caller actually held.
    pub(crate) fn add_borrow_bytes(&mut self, bytes: u64) {
        self.borrow_bytes = self.borrow_bytes.saturating_add(bytes);
    }

    /// Add source passes.
    pub(crate) fn add_source_passes(&mut self, passes: u64) {
        self.source_passes = self.source_passes.saturating_add(passes);
    }

    /// Add fits.
    pub(crate) fn add_fit_count(&mut self, fits: u64) {
        self.fit_count = self.fit_count.saturating_add(fits);
    }

    /// Add tests.
    pub(crate) fn add_test_count(&mut self, tests: u64) {
        self.test_count = self.test_count.saturating_add(tests);
    }

    /// Add retained outputs.
    pub(crate) fn add_retained_outputs(&mut self, outputs: u64) {
        self.retained_outputs = self.retained_outputs.saturating_add(outputs);
    }

    /// Add spill bytes.
    pub(crate) fn add_spill_bytes(&mut self, bytes: u64) {
        self.spill_bytes = self.spill_bytes.saturating_add(bytes);
    }

    /// Record the worker-thread count for this repeat.
    pub(crate) fn set_threads(&mut self, threads: u32) {
        self.threads = threads;
    }

    /// Record a device name. An empty name stays on the host (`None`).
    pub(crate) fn set_device(&mut self, device: Option<&str>) {
        self.device = device.filter(|name| !name.is_empty()).map(str::to_owned);
    }

    /// Add nanoseconds for `name`, summing a repeated stage in this repeat.
    pub(crate) fn add_stage_time(&mut self, name: &str, nanos: u64) {
        if let Some(stage) = self.stage_times_ns.iter_mut().find(|stage| stage.name == name) {
            stage.nanos = stage.nanos.saturating_add(nanos);
            return;
        }
        self.stage_times_ns.push(StageTime { name: name.to_owned(), nanos });
    }

    /// Mark the repeat cancelled.
    pub(crate) fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Mark the repeat failed.
    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }

    /// Read peak RSS now. On hosts without a std-only probe this stores [`PeakRss::Unsupported`].
    pub(crate) fn observe_peak_rss(&mut self) {
        self.peak_rss = sample_peak_rss();
    }
}

/// Min, median, and max of one counter across repeats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CountVariability {
    /// Repeats that reported this counter.
    pub count: u64,
    /// Smallest value.
    pub min: u64,
    /// Floor median. Even counts use [`u64::midpoint`] of the two central values.
    pub median: u64,
    /// Largest value.
    pub max: u64,
}

impl CountVariability {
    /// Variability of `counts` in reported order. Empty input is all zeros.
    #[must_use]
    pub(crate) fn from_counts(counts: impl IntoIterator<Item = u64>) -> Self {
        let mut values: Vec<u64> = counts.into_iter().collect();
        if values.is_empty() {
            return Self { count: 0, min: 0, median: 0, max: 0 };
        }
        values.sort_unstable();
        let last = values.len() - 1;
        let mid = values.len() / 2;
        let median =
            if values.len() % 2 == 1 { values[mid] } else { values[mid - 1].midpoint(values[mid]) };
        Self {
            count: u64::try_from(values.len()).unwrap_or(u64::MAX),
            min: values[0],
            median,
            max: values[last],
        }
    }
}

/// One stage's time variability across repeats that named it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StageVariability {
    /// Stage id.
    pub name: String,
    /// Min / median / max nanoseconds.
    pub nanos: CountVariability,
}

/// Peak RSS across repeats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PeakRssSummary {
    /// Every repeat reported a byte count.
    Bytes(CountVariability),
    /// At least one repeat had no sample, or there were no repeats.
    Unsupported {
        /// Why the summary has no peak-RSS distribution.
        reason: &'static str,
    },
}

/// Min / median / max across repeats. Flags are set if any repeat set them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasurementSummary {
    /// Repeat count. Zero when `samples` was empty.
    pub repeats: u64,
    /// Peak RSS distribution, or why it is absent.
    pub peak_rss: PeakRssSummary,
    /// Allocation bytes.
    pub allocation_bytes: CountVariability,
    /// Copy bytes.
    pub copy_bytes: CountVariability,
    /// Borrow bytes.
    pub borrow_bytes: CountVariability,
    /// Source passes.
    pub source_passes: CountVariability,
    /// Fits.
    pub fit_count: CountVariability,
    /// Tests.
    pub test_count: CountVariability,
    /// Retained outputs.
    pub retained_outputs: CountVariability,
    /// Spill bytes.
    pub spill_bytes: CountVariability,
    /// Reported thread counts.
    pub threads: CountVariability,
    /// Device name when every repeat named that same device.
    /// Host-only and mixed repeats are `None`.
    pub device: Option<String>,
    /// Stages in name order. A stage missing from a repeat is omitted, not treated as zero.
    pub stage_times_ns: Vec<StageVariability>,
    /// Any repeat was cancelled.
    pub cancelled: bool,
    /// Any repeat failed.
    pub failed: bool,
}

impl MeasurementSummary {
    /// Reduce `samples`. Order does not matter. Flags OR across repeats.
    #[must_use]
    pub(crate) fn from_repeats(samples: &[MeasurementRecord]) -> Self {
        Self {
            repeats: u64::try_from(samples.len()).unwrap_or(u64::MAX),
            peak_rss: peak_variability(samples),
            allocation_bytes: counts(samples, |sample| sample.allocation_bytes),
            copy_bytes: counts(samples, |sample| sample.copy_bytes),
            borrow_bytes: counts(samples, |sample| sample.borrow_bytes),
            source_passes: counts(samples, |sample| sample.source_passes),
            fit_count: counts(samples, |sample| sample.fit_count),
            test_count: counts(samples, |sample| sample.test_count),
            retained_outputs: counts(samples, |sample| sample.retained_outputs),
            spill_bytes: counts(samples, |sample| sample.spill_bytes),
            threads: counts(samples, |sample| u64::from(sample.threads)),
            device: shared_device(samples),
            stage_times_ns: stage_variability(samples),
            cancelled: samples.iter().any(|sample| sample.cancelled),
            failed: samples.iter().any(|sample| sample.failed),
        }
    }
}

fn counts(
    samples: &[MeasurementRecord],
    counter: impl Fn(&MeasurementRecord) -> u64,
) -> CountVariability {
    CountVariability::from_counts(samples.iter().map(counter))
}

fn shared_device(samples: &[MeasurementRecord]) -> Option<String> {
    let first = samples.first()?.device.as_deref()?;
    samples.iter().all(|sample| sample.device.as_deref() == Some(first)).then(|| first.to_owned())
}

fn stage_variability(samples: &[MeasurementRecord]) -> Vec<StageVariability> {
    let mut names: Vec<&str> = samples
        .iter()
        .flat_map(|sample| sample.stage_times_ns.iter().map(|stage| stage.name.as_str()))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
        .into_iter()
        .map(|name| StageVariability {
            name: name.to_owned(),
            nanos: CountVariability::from_counts(samples.iter().filter_map(|sample| {
                sample
                    .stage_times_ns
                    .iter()
                    .find(|stage| stage.name == name)
                    .map(|stage| stage.nanos)
            })),
        })
        .collect()
}

fn peak_variability(samples: &[MeasurementRecord]) -> PeakRssSummary {
    if samples.is_empty() {
        return PeakRssSummary::Unsupported { reason: PEAK_RSS_NO_REPEATS };
    }
    let mut reason: Option<&'static str> = None;
    let mut bytes = Vec::new();
    for sample in samples {
        match sample.peak_rss {
            PeakRss::Bytes(n) => bytes.push(n),
            PeakRss::Unsupported { reason: sample_reason } => {
                reason = Some(match reason {
                    Some(seen) if seen != sample_reason => PEAK_RSS_INCOMPLETE,
                    Some(seen) => seen,
                    None => sample_reason,
                });
            }
        }
    }
    if bytes.len() == samples.len() {
        PeakRssSummary::Bytes(CountVariability::from_counts(bytes))
    } else if bytes.is_empty() {
        PeakRssSummary::Unsupported { reason: reason.unwrap_or(PEAK_RSS_INCOMPLETE) }
    } else {
        PeakRssSummary::Unsupported { reason: PEAK_RSS_INCOMPLETE }
    }
}

/// Peak RSS for this process, or an explicit unsupported reason.
///
/// Linux: `VmHWM` in `/proc/self/status`. Elsewhere: [`PEAK_RSS_NEEDS_LIBC`].
#[must_use]
pub(crate) fn sample_peak_rss() -> PeakRss {
    #[cfg(target_os = "linux")]
    {
        sample_linux_vm_hwm()
    }
    #[cfg(not(target_os = "linux"))]
    {
        PeakRss::Unsupported { reason: PEAK_RSS_NEEDS_LIBC }
    }
}

#[cfg(target_os = "linux")]
fn sample_linux_vm_hwm() -> PeakRss {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return PeakRss::Unsupported { reason: PEAK_RSS_PROC_UNREADABLE };
    };
    match vm_hwm_bytes(&text) {
        Some(bytes) => PeakRss::Bytes(bytes),
        None => PeakRss::Unsupported { reason: PEAK_RSS_VMHWM_MISSING },
    }
}

/// Parse `VmHWM` as bytes. The kernel prints kibibytes (`kB`).
fn vm_hwm_bytes(status: &str) -> Option<u64> {
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmHWM:") else { continue };
        let mut parts = rest.split_whitespace();
        let kibibytes: u64 = parts.next()?.parse().ok()?;
        if parts.next() != Some("kB") {
            return None;
        }
        return kibibytes.checked_mul(1024);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_work_variability_cancellation_and_failure() {
        let mut recorded = MeasurementRecord::default();
        assert_eq!(recorded.allocation_bytes, 0);
        recorded.observe_peak_rss();
        assert_eq!(recorded.allocation_bytes, 0, "peak RSS sampling is not an allocator hook");
        if let PeakRss::Unsupported { reason } = recorded.peak_rss {
            assert!(!reason.is_empty());
        }

        let native = vec![0u64; 32];
        let bytes = u64::try_from(std::mem::size_of_val(native.as_slice())).unwrap();
        recorded.add_allocation_bytes(bytes);
        assert_eq!(recorded.allocation_bytes, bytes);
        assert_eq!(bytes, 256);

        let mut fast = MeasurementRecord::default();
        fast.add_allocation_bytes(10);
        fast.add_copy_bytes(1);
        fast.add_borrow_bytes(2);
        fast.add_source_passes(1);
        fast.add_fit_count(1);
        fast.add_test_count(1);
        fast.add_retained_outputs(1);
        fast.add_spill_bytes(4);
        fast.set_threads(1);
        fast.set_device(Some("gpu0"));
        fast.add_stage_time("estimate", 100);
        fast.fail();

        let mut slow = MeasurementRecord::default();
        slow.add_allocation_bytes(30);
        slow.add_copy_bytes(3);
        slow.add_borrow_bytes(6);
        slow.add_source_passes(3);
        slow.add_fit_count(3);
        slow.add_test_count(3);
        slow.add_retained_outputs(3);
        slow.add_spill_bytes(8);
        slow.set_threads(3);
        slow.set_device(Some("gpu0"));
        slow.add_stage_time("estimate", 300);
        slow.cancel();

        assert!(fast.failed);
        assert!(!fast.cancelled);
        assert!(slow.cancelled);
        assert!(!slow.failed);

        let summary = MeasurementSummary::from_repeats(&[fast, slow]);
        assert_eq!(summary.repeats, 2);
        assert!(matches!(
            summary.peak_rss,
            PeakRssSummary::Unsupported { reason: PEAK_RSS_NOT_SAMPLED }
        ));
        assert_eq!(
            summary.allocation_bytes,
            CountVariability { count: 2, min: 10, median: 20, max: 30 }
        );
        assert_eq!(summary.copy_bytes, CountVariability { count: 2, min: 1, median: 2, max: 3 });
        assert_eq!(summary.borrow_bytes, CountVariability { count: 2, min: 2, median: 4, max: 6 });
        assert_eq!(summary.source_passes, CountVariability { count: 2, min: 1, median: 2, max: 3 });
        assert_eq!(summary.fit_count, CountVariability { count: 2, min: 1, median: 2, max: 3 });
        assert_eq!(summary.test_count, CountVariability { count: 2, min: 1, median: 2, max: 3 });
        assert_eq!(
            summary.retained_outputs,
            CountVariability { count: 2, min: 1, median: 2, max: 3 }
        );
        assert_eq!(summary.spill_bytes, CountVariability { count: 2, min: 4, median: 6, max: 8 });
        assert_eq!(summary.threads, CountVariability { count: 2, min: 1, median: 2, max: 3 });
        assert_eq!(summary.device.as_deref(), Some("gpu0"));
        assert_eq!(
            summary.stage_times_ns,
            vec![StageVariability {
                name: "estimate".to_owned(),
                nanos: CountVariability { count: 2, min: 100, median: 200, max: 300 },
            }]
        );
        assert!(summary.cancelled);
        assert!(summary.failed);
    }

    #[test]
    fn flags_stay_on_the_record_they_were_set_on() {
        let mut record = MeasurementRecord::default();
        record.cancel();
        record.fail();
        let summary = MeasurementSummary::from_repeats(std::slice::from_ref(&record));
        assert!(record.cancelled);
        assert!(record.failed);
        assert!(summary.cancelled);
        assert!(summary.failed);
    }

    #[test]
    fn linux_vm_hwm_line_is_kibibytes() {
        let text = "VmPeak:\t8 kB\nVmHWM:\t3 kB\nVmRSS:\t2 kB\n";
        assert_eq!(vm_hwm_bytes(text), Some(3072));
        assert_eq!(vm_hwm_bytes("VmRSS:\t2 kB\n"), None);
        assert_eq!(vm_hwm_bytes("VmHWM:\t3 MB\n"), None);
    }
}
