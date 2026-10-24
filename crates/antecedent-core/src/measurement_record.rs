//! Counters for one native workload repeat, read from the engine's own instrumentation.
//!
//! [`measure`] runs a workload under [`observe_with_context`] and builds the record from
//! what the engine observed at its source boundaries: scans, copied bytes, model fits,
//! conditional-independence tests, retained bytes, spill, admitted threads and the
//! completion status ([`MeasurementRecord::from_execution_facts`]). A [`ResourceProbe`]
//! adds the figures only the process can supply: allocator bytes and counts across the
//! run, the live-byte high-water mark, and peak resident set size. Fields a run did not
//! observe are named in [`MeasurementRecord::unobserved`] instead of reading as zero.
//! Facts that a caller declared, rather than the engine emitting them, are never read.
//!
//! [`HostProbe`] has no allocator hook: `antecedent-core` forbids `unsafe_code`, so the
//! counting allocator and the `getrusage` peak-RSS reader live in `antecedent-kernels`
//! (`native_probe`). On Linux [`HostProbe`] still reads `VmHWM` from `/proc/self/status`
//! with `std::fs` alone; other hosts report [`PeakRss::Unsupported`] here.
//!
//! [`MeasurementSummary`] reduces repeats to min / median / max, matching the median
//! summaries in the bench reports. Even-count medians use [`u64::midpoint`] (floor of the
//! two central values). [`MeasurementRecord::to_json`] and
//! [`MeasurementSummary::to_json`] write the optional `measurement` object
//! `scripts/bench_regression.py` attaches when `<bench>/measurement.json` is present.
//! This module is not a second benchmark runner.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::time::Instant;

use crate::execution::ExecutionContext;
use crate::execution_attempt::{ExecutionAttempt, observe_with_context};
use crate::execution_facts::{ExecutionFactStatus, ExecutionFacts, OutputRetention};

/// Not sampled until [`MeasurementRecord::observe_peak_rss`] or an explicit set.
pub const PEAK_RSS_NOT_SAMPLED: &str =
    "peak RSS was not sampled; report it explicitly or call observe_peak_rss";

/// `getrusage` needs `unsafe`, which this crate forbids; use the probe in `antecedent-kernels`.
pub const PEAK_RSS_NEEDS_LIBC: &str = concat!(
    "peak RSS is unsupported in antecedent-core on this host: reading it needs getrusage ",
    "(RUSAGE_SELF ru_maxrss), which is unsafe and this crate forbids unsafe_code. Use ",
    "antecedent_kernels::native_probe::NativeProbe. Linux VmHWM from /proc/self/status ",
    "is the std-only path"
);

/// `/proc/self/status` did not contain a `VmHWM` kibibyte field.
#[cfg(target_os = "linux")]
pub const PEAK_RSS_VMHWM_MISSING: &str =
    "peak RSS is unsupported: /proc/self/status has no VmHWM kibibyte field";

/// `/proc/self/status` could not be read.
#[cfg(target_os = "linux")]
pub const PEAK_RSS_PROC_UNREADABLE: &str =
    "peak RSS is unsupported: /proc/self/status could not be read";

/// Repeats disagreed, or only some of them had a sample.
pub const PEAK_RSS_INCOMPLETE: &str = "peak RSS was not available for every repeat";

/// [`MeasurementSummary::from_repeats`] on an empty slice.
pub const PEAK_RSS_NO_REPEATS: &str = "no measurement repeats";

/// Peak resident set, or why this process cannot read it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeakRss {
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
pub struct StageTime {
    /// Stage id, for example `estimate`.
    pub name: String,
    /// Nanoseconds reported for this stage in the repeat.
    pub nanos: u64,
}

/// Counters one native repeat reports. Zero means none reported, not "unknown".
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeasurementRecord {
    /// Peak RSS sample for this repeat.
    pub peak_rss: PeakRss,
    /// Bytes the allocator handed out across the run. Needs a [`ResourceProbe`] with a
    /// counting allocator.
    pub allocation_bytes: u64,
    /// Allocation calls across the run.
    pub allocation_count: u64,
    /// Highest live heap bytes above the level at the start of the run.
    pub peak_live_bytes: u64,
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
    /// Admitted output bytes retained at handoff. Not heap usage and not RSS.
    pub retained_bytes: u64,
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
    /// Fields whose source boundary the run did not observe. A zero in these is not a
    /// measurement.
    pub unobserved: Vec<&'static str>,
}

/// Every field a run can leave unobserved, in record order.
pub const OBSERVABLE_FIELDS: [&str; 9] = [
    "allocation_bytes",
    "copy_bytes",
    "borrow_bytes",
    "source_passes",
    "fit_count",
    "test_count",
    "retained_bytes",
    "spill_bytes",
    "threads",
];

/// Allocator totals read from a [`ResourceProbe`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationSample {
    /// Bytes handed out since the allocator started counting.
    pub total_bytes: u64,
    /// Allocation calls since the allocator started counting.
    pub count: u64,
    /// Heap bytes live now.
    pub live_bytes: u64,
    /// Highest live heap bytes since the last [`ResourceProbe::begin_window`].
    pub peak_live_bytes: u64,
}

/// Process-level figures only a probe can read: allocator state and resident set size.
pub trait ResourceProbe {
    /// Reset the live-byte high-water mark to the current level.
    fn begin_window(&self) {}
    /// Allocator totals, or `None` when no counting allocator is installed.
    fn allocation(&self) -> Option<AllocationSample>;
    /// Peak resident set size of this process so far.
    fn peak_rss(&self) -> PeakRss;
}

/// The probe available to this crate: no allocator hook, peak RSS from `VmHWM` on Linux.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostProbe;

impl ResourceProbe for HostProbe {
    fn allocation(&self) -> Option<AllocationSample> {
        None
    }

    fn peak_rss(&self) -> PeakRss {
        sample_peak_rss()
    }
}

impl MeasurementRecord {
    /// Read a record from engine-emitted facts. Facts a caller declared are not read:
    /// every observable field is then listed as unobserved.
    #[must_use]
    pub fn from_execution_facts(facts: &ExecutionFacts) -> Self {
        let mut record = Self::default();
        if !facts.is_native() {
            record.unobserved.extend(OBSERVABLE_FIELDS);
            return record;
        }
        match facts.observed_scans() {
            Some(scans) => record.source_passes = scans,
            None => record.unobserved.push("source_passes"),
        }
        match facts.observed_copied_bytes() {
            Some(bytes) => record.copy_bytes = bytes,
            None => record.unobserved.push("copy_bytes"),
        }
        match facts.observed_model_fits() {
            Some(fits) => record.fit_count = fits,
            None => record.unobserved.push("fit_count"),
        }
        // Bit 8 is `PhysicalEvent::ConditionalIndependenceTest`.
        if facts.native_coverage().is_some_and(|mask| mask & (1 << 8) != 0) {
            record.test_count = facts.counts().tests();
        } else {
            record.unobserved.push("test_count");
        }
        match facts.observed_retained_bytes() {
            Some(bytes) => record.retained_bytes = bytes,
            None => record.unobserved.push("retained_bytes"),
        }
        match facts.observed_spill_bytes() {
            Some(bytes) => record.spill_bytes = bytes,
            None => record.unobserved.push("spill_bytes"),
        }
        match facts.peak_admitted_threads() {
            Some(threads) => record.threads = u32::try_from(threads).unwrap_or(u32::MAX),
            None => record.unobserved.push("threads"),
        }
        if facts.retention() != OutputRetention::Unmeasured {
            record.retained_outputs = 1;
        }
        // The facts carry no borrow figure; Arrow ingest reports it separately.
        record.unobserved.push("borrow_bytes");
        record.unobserved.push("allocation_bytes");
        match facts.status() {
            ExecutionFactStatus::Completed => {}
            ExecutionFactStatus::Cancelled => record.cancelled = true,
            _ => record.failed = true,
        }
        record
    }

    /// Fold the allocator movement between two samples into the record and clear the
    /// allocation fields from [`Self::unobserved`].
    pub fn add_allocation_window(&mut self, before: AllocationSample, after: AllocationSample) {
        self.allocation_bytes = self
            .allocation_bytes
            .saturating_add(after.total_bytes.saturating_sub(before.total_bytes));
        self.allocation_count =
            self.allocation_count.saturating_add(after.count.saturating_sub(before.count));
        self.peak_live_bytes =
            self.peak_live_bytes.max(after.peak_live_bytes.saturating_sub(before.live_bytes));
        self.unobserved.retain(|field| *field != "allocation_bytes");
    }

    /// Whether `field` (a name from [`OBSERVABLE_FIELDS`]) was observed in this run.
    #[must_use]
    pub fn observed(&self, field: &str) -> bool {
        !self.unobserved.iter().any(|name| *name == field)
    }

    /// Add bytes of a native allocation the caller actually performed.
    pub fn add_allocation_bytes(&mut self, bytes: u64) {
        self.allocation_bytes = self.allocation_bytes.saturating_add(bytes);
    }

    /// Add bytes of a native copy the caller actually performed.
    pub fn add_copy_bytes(&mut self, bytes: u64) {
        self.copy_bytes = self.copy_bytes.saturating_add(bytes);
    }

    /// Add bytes of a native borrow the caller actually held.
    pub fn add_borrow_bytes(&mut self, bytes: u64) {
        self.borrow_bytes = self.borrow_bytes.saturating_add(bytes);
        self.unobserved.retain(|field| *field != "borrow_bytes");
    }

    /// Add source passes.
    pub fn add_source_passes(&mut self, passes: u64) {
        self.source_passes = self.source_passes.saturating_add(passes);
    }

    /// Add fits.
    pub fn add_fit_count(&mut self, fits: u64) {
        self.fit_count = self.fit_count.saturating_add(fits);
    }

    /// Add tests.
    pub fn add_test_count(&mut self, tests: u64) {
        self.test_count = self.test_count.saturating_add(tests);
    }

    /// Add retained outputs.
    pub fn add_retained_outputs(&mut self, outputs: u64) {
        self.retained_outputs = self.retained_outputs.saturating_add(outputs);
    }

    /// Add spill bytes.
    pub fn add_spill_bytes(&mut self, bytes: u64) {
        self.spill_bytes = self.spill_bytes.saturating_add(bytes);
    }

    /// Record the worker-thread count for this repeat.
    pub fn set_threads(&mut self, threads: u32) {
        self.threads = threads;
    }

    /// Record a device name. An empty name stays on the host (`None`).
    pub fn set_device(&mut self, device: Option<&str>) {
        self.device = device.filter(|name| !name.is_empty()).map(str::to_owned);
    }

    /// Add nanoseconds for `name`, summing a repeated stage in this repeat.
    pub fn add_stage_time(&mut self, name: &str, nanos: u64) {
        if let Some(stage) = self.stage_times_ns.iter_mut().find(|stage| stage.name == name) {
            stage.nanos = stage.nanos.saturating_add(nanos);
            return;
        }
        self.stage_times_ns.push(StageTime { name: name.to_owned(), nanos });
    }

    /// Mark the repeat cancelled.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Mark the repeat failed.
    pub fn fail(&mut self) {
        self.failed = true;
    }

    /// Read peak RSS now. On hosts without a std-only probe this stores [`PeakRss::Unsupported`].
    pub fn observe_peak_rss(&mut self) {
        self.peak_rss = sample_peak_rss();
    }
}

/// Min, median, and max of one counter across repeats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountVariability {
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
    pub fn from_counts(counts: impl IntoIterator<Item = u64>) -> Self {
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
pub struct StageVariability {
    /// Stage id.
    pub name: String,
    /// Min / median / max nanoseconds.
    pub nanos: CountVariability,
}

/// Peak RSS across repeats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeakRssSummary {
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
pub struct MeasurementSummary {
    /// Repeat count. Zero when `samples` was empty.
    pub repeats: u64,
    /// Peak RSS distribution, or why it is absent.
    pub peak_rss: PeakRssSummary,
    /// Allocation bytes.
    pub allocation_bytes: CountVariability,
    /// Allocation calls.
    pub allocation_count: CountVariability,
    /// Live-byte high-water mark above the level at the start of each repeat.
    pub peak_live_bytes: CountVariability,
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
    /// Admitted retained bytes.
    pub retained_bytes: CountVariability,
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
    /// Fields that at least one repeat did not observe, in [`OBSERVABLE_FIELDS`] order.
    pub unobserved: Vec<&'static str>,
}

impl MeasurementSummary {
    /// Reduce `samples`. Order does not matter. Flags OR across repeats.
    #[must_use]
    pub fn from_repeats(samples: &[MeasurementRecord]) -> Self {
        Self {
            repeats: u64::try_from(samples.len()).unwrap_or(u64::MAX),
            peak_rss: peak_variability(samples),
            allocation_bytes: counts(samples, |sample| sample.allocation_bytes),
            allocation_count: counts(samples, |sample| sample.allocation_count),
            peak_live_bytes: counts(samples, |sample| sample.peak_live_bytes),
            copy_bytes: counts(samples, |sample| sample.copy_bytes),
            borrow_bytes: counts(samples, |sample| sample.borrow_bytes),
            source_passes: counts(samples, |sample| sample.source_passes),
            fit_count: counts(samples, |sample| sample.fit_count),
            test_count: counts(samples, |sample| sample.test_count),
            retained_outputs: counts(samples, |sample| sample.retained_outputs),
            retained_bytes: counts(samples, |sample| sample.retained_bytes),
            spill_bytes: counts(samples, |sample| sample.spill_bytes),
            threads: counts(samples, |sample| u64::from(sample.threads)),
            device: shared_device(samples),
            stage_times_ns: stage_variability(samples),
            cancelled: samples.iter().any(|sample| sample.cancelled),
            failed: samples.iter().any(|sample| sample.failed),
            unobserved: OBSERVABLE_FIELDS
                .into_iter()
                .filter(|field| samples.iter().any(|sample| !sample.observed(field)))
                .collect(),
        }
    }
}

/// Run `work` under the engine's execution observer and read what it actually did.
///
/// The record holds the engine's own counts for the work (see
/// [`MeasurementRecord::from_execution_facts`]), the allocator movement and peak
/// resident set from `probe`, and the wall time of the work as stage `total`. The
/// result of `work` is returned unchanged, including an error.
pub fn measure<T, E>(
    ctx: &ExecutionContext,
    probe: &dyn ResourceProbe,
    work: impl FnOnce() -> Result<T, E>,
) -> (Result<T, E>, MeasurementRecord) {
    probe.begin_window();
    let before = probe.allocation();
    let started = Instant::now();
    let ExecutionAttempt { result, facts, .. } = observe_with_context(ctx, work);
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let after = probe.allocation();
    let mut record = facts.as_ref().map_or_else(
        || {
            let mut unread = MeasurementRecord::default();
            unread.unobserved.extend(OBSERVABLE_FIELDS);
            unread
        },
        MeasurementRecord::from_execution_facts,
    );
    if let (Some(before), Some(after)) = (before, after) {
        record.add_allocation_window(before, after);
    }
    record.add_stage_time("total", elapsed);
    record.peak_rss = probe.peak_rss();
    (result, record)
}

fn push_count(out: &mut String, key: &str, value: u64) {
    let _ = write!(out, "\"{key}\":{value},");
}

fn push_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_stages(out: &mut String, stages: impl Iterator<Item = (String, u64)>) {
    out.push_str("\"stage_times_ns\":[");
    for (index, (name, nanos)) in stages.enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":");
        push_string(out, &name);
        let _ = write!(out, ",\"nanos\":{nanos}}}");
    }
    out.push_str("],");
}

fn push_peak_rss(out: &mut String, peak: PeakRss) {
    match peak {
        PeakRss::Bytes(bytes) => push_count(out, "peak_rss_bytes", bytes),
        PeakRss::Unsupported { reason } => {
            out.push_str("\"peak_rss_unsupported\":");
            push_string(out, reason);
            out.push(',');
        }
    }
}

fn push_device(out: &mut String, device: Option<&str>) {
    out.push_str("\"device\":");
    match device {
        Some(name) => push_string(out, name),
        None => out.push_str("null"),
    }
    out.push(',');
}

fn push_unobserved(out: &mut String, fields: &[&'static str]) {
    out.push_str("\"unobserved\":[");
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        push_string(out, field);
    }
    out.push(']');
}

impl MeasurementRecord {
    /// The `measurement` object `scripts/bench_regression.py` reads, for one repeat.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        push_peak_rss(&mut out, self.peak_rss);
        push_count(&mut out, "allocation_bytes", self.allocation_bytes);
        push_count(&mut out, "allocation_count", self.allocation_count);
        push_count(&mut out, "peak_live_bytes", self.peak_live_bytes);
        push_count(&mut out, "copy_bytes", self.copy_bytes);
        push_count(&mut out, "borrow_bytes", self.borrow_bytes);
        push_count(&mut out, "source_passes", self.source_passes);
        push_count(&mut out, "fit_count", self.fit_count);
        push_count(&mut out, "test_count", self.test_count);
        push_count(&mut out, "retained_outputs", self.retained_outputs);
        push_count(&mut out, "retained_bytes", self.retained_bytes);
        push_count(&mut out, "spill_bytes", self.spill_bytes);
        push_count(&mut out, "threads", u64::from(self.threads));
        push_device(&mut out, self.device.as_deref());
        push_stages(&mut out, self.stage_times_ns.iter().map(|s| (s.name.clone(), s.nanos)));
        let _ = write!(out, "\"cancelled\":{},\"failed\":{},", self.cancelled, self.failed);
        push_unobserved(&mut out, &self.unobserved);
        out.push('}');
        out
    }
}

impl MeasurementSummary {
    /// The `measurement` object for a set of repeats: each counter is its median and
    /// `variability` carries count / min / median / max. Peak RSS is the median when
    /// every repeat had a sample, else the unsupported reason.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        match self.peak_rss {
            PeakRssSummary::Bytes(v) => push_count(&mut out, "peak_rss_bytes", v.median),
            PeakRssSummary::Unsupported { reason } => {
                push_peak_rss(&mut out, PeakRss::Unsupported { reason });
            }
        }
        let counters = self.counters();
        for (name, v) in &counters {
            push_count(&mut out, name, v.median);
        }
        push_device(&mut out, self.device.as_deref());
        push_stages(
            &mut out,
            self.stage_times_ns.iter().map(|stage| (stage.name.clone(), stage.nanos.median)),
        );
        let _ = write!(out, "\"cancelled\":{},\"failed\":{},", self.cancelled, self.failed);
        out.push_str("\"variability\":{");
        let mut named: Vec<(&str, CountVariability)> = counters;
        if let PeakRssSummary::Bytes(v) = self.peak_rss {
            named.insert(0, ("peak_rss_bytes", v));
        }
        for (index, (name, v)) in named.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "\"{name}\":{{\"count\":{},\"min\":{},\"median\":{},\"max\":{}}}",
                v.count, v.min, v.median, v.max
            );
        }
        out.push_str("},");
        push_unobserved(&mut out, &self.unobserved);
        out.push('}');
        out
    }

    fn counters(&self) -> Vec<(&'static str, CountVariability)> {
        vec![
            ("allocation_bytes", self.allocation_bytes),
            ("allocation_count", self.allocation_count),
            ("peak_live_bytes", self.peak_live_bytes),
            ("copy_bytes", self.copy_bytes),
            ("borrow_bytes", self.borrow_bytes),
            ("source_passes", self.source_passes),
            ("fit_count", self.fit_count),
            ("test_count", self.test_count),
            ("retained_outputs", self.retained_outputs),
            ("retained_bytes", self.retained_bytes),
            ("spill_bytes", self.spill_bytes),
            ("threads", self.threads),
        ]
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
pub fn sample_peak_rss() -> PeakRss {
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
#[cfg(any(target_os = "linux", test))]
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
    use std::cell::Cell;

    use super::*;
    use crate::execution::KernelPolicy;
    use crate::execution_attempt::{PhysicalEvent, record_physical};
    use crate::execution_facts::{
        ExecutionBackend, ExecutionCounts, ExecutionResources, MeasuredExecution,
    };

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(1)
    }

    /// A workload that builds `buffers` owned buffers of `rows` f64 values and reports each
    /// scan and copy at the boundary where it happens, as the instrumented engine code does.
    /// Returns the bytes it actually copied.
    fn copy_workload(rows: usize, buffers: usize) -> Result<u64, &'static str> {
        let source: Vec<f64> = (0..rows).map(|i| i as f64).collect();
        let mut copied = 0u64;
        for _ in 0..buffers {
            record_physical(PhysicalEvent::Scan, 1);
            let owned = source.clone();
            let bytes = u64::try_from(std::mem::size_of_val(owned.as_slice())).unwrap();
            record_physical(PhysicalEvent::Copy, 1);
            record_physical(PhysicalEvent::CopiedBytes, bytes);
            copied += bytes;
        }
        Ok(copied)
    }

    #[test]
    fn the_record_holds_the_work_the_run_performed() {
        let (result, record) = measure(&context(), &HostProbe, || copy_workload(100, 3));
        let copied = result.unwrap();
        assert_eq!(copied, 3 * 100 * 8);
        assert_eq!(record.copy_bytes, copied);
        assert_eq!(record.source_passes, 3);
        assert!(record.observed("copy_bytes"));
        assert!(record.observed("source_passes"));
        assert!(!record.failed && !record.cancelled);
        assert!(
            record.stage_times_ns.iter().any(|stage| stage.name == "total"),
            "the wall time of the work is a stage of the record"
        );
    }

    #[test]
    fn counters_move_with_the_work_and_boundaries_never_reached_stay_unobserved() {
        let (_, small) = measure(&context(), &HostProbe, || copy_workload(50, 2));
        let (_, large) = measure(&context(), &HostProbe, || copy_workload(200, 6));
        assert_eq!(small.copy_bytes, 2 * 50 * 8);
        assert_eq!(large.copy_bytes, 6 * 200 * 8);
        assert_eq!((small.source_passes, large.source_passes), (2, 6));
        for record in [&small, &large] {
            // Nothing in the workload fits a model, runs a test, or spills.
            assert!(!record.observed("fit_count"));
            assert!(!record.observed("test_count"));
            assert!(!record.observed("spill_bytes"));
            // No counting allocator sits behind `HostProbe`.
            assert!(!record.observed("allocation_bytes"));
            assert_eq!(record.allocation_bytes, 0);
        }
    }

    #[test]
    fn a_failing_workload_keeps_its_error_and_the_work_it_did() {
        let (result, record) = measure(&context(), &HostProbe, || {
            copy_workload(10, 1)?;
            Err::<(), _>("refused")
        });
        assert_eq!(result, Err("refused"));
        assert!(record.failed);
        assert_eq!(record.copy_bytes, 80);
        assert_eq!(record.source_passes, 1);
    }

    #[test]
    fn cancellation_is_read_from_the_context() {
        let ctx = context();
        let (result, record) = measure(&ctx, &HostProbe, || {
            ctx.cancellation.cancel();
            copy_workload(10, 1)
        });
        assert!(result.is_ok());
        assert!(record.cancelled);
        assert!(!record.failed);
    }

    #[test]
    fn declared_facts_are_not_read_as_observations() {
        let measured = MeasuredExecution::new(
            ExecutionCounts::new(9, 9, 9, 9, 9),
            OutputRetention::FullReusable,
            ExecutionResources::new(4, 1 << 20, 7),
            ExecutionBackend::new("declared".into(), KernelPolicy::default_policy()),
            ExecutionFactStatus::Completed,
        )
        .with_observed_buffer_bytes(Some(512), Some(512));
        for facts in [
            ExecutionFacts::new(measured.clone(), None, false),
            ExecutionFacts::from_artifact(measured, u64::MAX),
        ] {
            let record = MeasurementRecord::from_execution_facts(&facts);
            assert_eq!(record.source_passes, 0);
            assert_eq!(record.copy_bytes, 0);
            assert_eq!(record.fit_count, 0);
            assert_eq!(record.spill_bytes, 0);
            assert_eq!(record.threads, 0);
            assert_eq!(record.unobserved, OBSERVABLE_FIELDS.to_vec());
        }
    }

    struct ScriptedProbe {
        samples: [AllocationSample; 2],
        reads: Cell<usize>,
        windows: Cell<u32>,
    }

    impl ResourceProbe for ScriptedProbe {
        fn begin_window(&self) {
            self.windows.set(self.windows.get() + 1);
        }

        fn allocation(&self) -> Option<AllocationSample> {
            let read = self.reads.get();
            self.reads.set(read + 1);
            Some(self.samples[read.min(1)])
        }

        fn peak_rss(&self) -> PeakRss {
            PeakRss::Bytes(4096)
        }
    }

    #[test]
    fn allocator_movement_is_the_difference_across_the_run() {
        let probe = ScriptedProbe {
            samples: [
                AllocationSample {
                    total_bytes: 1_000,
                    count: 10,
                    live_bytes: 600,
                    peak_live_bytes: 600,
                },
                AllocationSample {
                    total_bytes: 5_000,
                    count: 25,
                    live_bytes: 700,
                    peak_live_bytes: 2_600,
                },
            ],
            reads: Cell::new(0),
            windows: Cell::new(0),
        };
        let (_, record) = measure(&context(), &probe, || copy_workload(8, 1));
        assert_eq!(probe.windows.get(), 1);
        assert_eq!(record.allocation_bytes, 4_000);
        assert_eq!(record.allocation_count, 15);
        assert_eq!(record.peak_live_bytes, 2_000, "peak above the level at the start");
        assert!(record.observed("allocation_bytes"));
        assert_eq!(record.peak_rss, PeakRss::Bytes(4096));
    }

    #[test]
    fn repeats_reduce_to_the_spread_of_the_observed_work() {
        let sizes = [(40usize, 1usize), (100, 2), (260, 5)];
        let records: Vec<MeasurementRecord> = sizes
            .iter()
            .map(|&(rows, buffers)| {
                measure(&context(), &HostProbe, || copy_workload(rows, buffers)).1
            })
            .collect();
        let expected: Vec<u64> =
            sizes.iter().map(|&(rows, buffers)| (rows * buffers * 8) as u64).collect();
        let summary = MeasurementSummary::from_repeats(&records);
        assert_eq!(summary.repeats, 3);
        assert_eq!(
            summary.copy_bytes,
            CountVariability {
                count: 3,
                min: *expected.iter().min().unwrap(),
                median: expected[1],
                max: *expected.iter().max().unwrap(),
            }
        );
        assert_eq!(summary.source_passes, CountVariability { count: 3, min: 1, median: 2, max: 5 });
        assert!(summary.unobserved.contains(&"fit_count"));
        assert!(!summary.unobserved.contains(&"copy_bytes"));
        let stage = summary.stage_times_ns.iter().find(|stage| stage.name == "total").unwrap();
        assert_eq!(stage.nanos.count, 3);
        let json = summary.to_json();
        assert!(json.contains("\"copy_bytes\":1600"), "{json}");
        assert!(json.contains("\"variability\""));
        assert!(json.contains("\"peak_rss_unsupported\"") || json.contains("\"peak_rss_bytes\""));
    }

    #[test]
    fn even_counts_take_the_floor_midpoint_and_empty_input_is_all_zero() {
        let even = CountVariability::from_counts([10, 31, 20, 40]);
        assert_eq!(even, CountVariability { count: 4, min: 10, median: 25, max: 40 });
        assert_eq!(
            CountVariability::from_counts([]),
            CountVariability { count: 0, min: 0, median: 0, max: 0 }
        );
        assert!(matches!(
            MeasurementSummary::from_repeats(&[]).peak_rss,
            PeakRssSummary::Unsupported { reason: PEAK_RSS_NO_REPEATS }
        ));
    }

    #[test]
    fn a_repeat_without_a_peak_rss_sample_withdraws_the_distribution() {
        let sampled = MeasurementRecord { peak_rss: PeakRss::Bytes(10), ..Default::default() };
        let unsampled = MeasurementRecord::default();
        assert!(matches!(
            MeasurementSummary::from_repeats(&[sampled.clone(), unsampled]).peak_rss,
            PeakRssSummary::Unsupported { reason: PEAK_RSS_INCOMPLETE }
        ));
        assert!(matches!(
            MeasurementSummary::from_repeats(&[sampled]).peak_rss,
            PeakRssSummary::Bytes(CountVariability { count: 1, min: 10, median: 10, max: 10 })
        ));
    }

    #[test]
    fn json_names_what_was_measured_and_escapes_text() {
        let (_, mut record) = measure(&context(), &HostProbe, || copy_workload(25, 2));
        record.set_device(Some("gpu \"0\""));
        record.add_borrow_bytes(64);
        let json = record.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"copy_bytes\":400,"), "{json}");
        assert!(json.contains("\"source_passes\":2,"));
        assert!(json.contains("\"borrow_bytes\":64,"));
        assert!(json.contains("\"device\":\"gpu \\\"0\\\"\""), "{json}");
        assert!(json.contains("\"unobserved\":["));
        assert!(record.observed("borrow_bytes"));
    }

    #[test]
    fn linux_vm_hwm_line_is_kibibytes() {
        let text = "VmPeak:\t8 kB\nVmHWM:\t3 kB\nVmRSS:\t2 kB\n";
        assert_eq!(vm_hwm_bytes(text), Some(3072));
        assert_eq!(vm_hwm_bytes("VmRSS:\t2 kB\n"), None);
        assert_eq!(vm_hwm_bytes("VmHWM:\t3 MB\n"), None);
    }
}
