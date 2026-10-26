//! The three workloads the rejected Polars adapter was meant to serve, run through the
//! accepted direct-Arrow preparation path: a wide adjusted effect, panel and
//! environment lag windows, and one shared design for many consumers.
//!
//! Each workload is measured for latency against a plain copy baseline, for peak
//! allocation with a counting allocator, and for identity across the Arrow batch,
//! Parquet and Arrow IPC routes. The packaging checks read the manifests that decide
//! what a wheel links. The declared budgets live in
//! `parity/polars_decision/direct_arrow_workloads.toml`; set
//! `ANTECEDENT_WORKLOAD_EVIDENCE` to a path to append the measured values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "test file: fixtures unwrap, and layout values are compared exactly"
)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::execution::AdmissionPool;
use antecedent_core::{ExecutionContext, KernelPolicy, VariableId};
use antecedent_data::design_layout::{DesignLayout, OwnershipCopy};
use antecedent_data::file_source::{ReadOptions, TableFile};
use antecedent_data::{
    GlobalId, LaggedFrame, MultiEnvironmentData, TableView, TabularData, TimeSeriesData,
    tabular_from_record_batch,
};
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

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

const F64_BYTES: u64 = 8;
const DECLARED: &str = include_str!("../../../parity/polars_decision/direct_arrow_workloads.toml");

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let dir = std::env::temp_dir()
            .join(format!("antecedent-direct-arrow-{}-{label}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A budget declared in the workload record: `key = value` under `[section]`.
fn declared(section: &str, key: &str) -> u64 {
    let mut inside = false;
    for line in DECLARED.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(header) = line.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
            inside = header == section;
        } else if inside {
            if let Some((name, value)) = line.split_once('=') {
                if name.trim() == key {
                    return value.trim().replace('_', "").parse().unwrap();
                }
            }
        }
    }
    panic!("[{section}] {key} is not declared in direct_arrow_workloads.toml")
}

fn record(workload: &str, key: &str, value: impl std::fmt::Display) {
    let Ok(path) = std::env::var("ANTECEDENT_WORKLOAD_EVIDENCE") else {
        return;
    };
    let mut file =
        std::fs::OpenOptions::new().create(true).append(true).open(path).expect("evidence file");
    writeln!(file, "{workload}.{key} = {value}").unwrap();
}

fn best_of<T>(runs: usize, mut work: impl FnMut() -> T) -> (T, Duration) {
    let mut best = Duration::MAX;
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        let out = work();
        best = best.min(start.elapsed());
        last = Some(out);
    }
    (last.expect("at least one run"), best)
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn batch(columns: &[(String, Vec<f64>)]) -> RecordBatch {
    let fields: Vec<Field> =
        columns.iter().map(|(name, _)| Field::new(name, DataType::Float64, false)).collect();
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|(_, values)| Arc::new(Float64Array::from(values.clone())) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn write_parquet(path: &Path, batch: &RecordBatch, rows_per_group: usize) {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_size(rows_per_group).build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn write_ipc(path: &Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
}

fn ids(data: &TabularData) -> Vec<VariableId> {
    data.schema().variables().iter().map(|variable| variable.id).collect()
}

fn read_file(path: &Path, names: &[String]) -> TabularData {
    let table = TableFile::open(path).unwrap();
    let plan = table.plan_columns(names, ReadOptions::with_chunk_rows(8192)).unwrap();
    let ctx = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(1, 1 << 30));
    table.materialize(&plan, &ctx).unwrap().into_data()
}

#[test]
fn wide_adjusted_effect_is_prepared_with_named_copies_and_one_identity_across_routes() {
    let rows = declared("wide_adjustment", "rows") as usize;
    let covariates = declared("wide_adjustment", "covariates") as usize;
    let mut next = lcg(11);
    let mut columns: Vec<(String, Vec<f64>)> =
        (0..covariates).map(|c| (format!("c{c}"), (0..rows).map(|_| next()).collect())).collect();
    columns.push(("t".into(), (0..rows).map(|_| f64::from(next() > 0.5)).collect()));
    columns.push(("y".into(), (0..rows).map(|_| next()).collect()));
    let record_batch = batch(&columns);
    let width = columns.len();
    let input_bytes = (rows * width) as u64 * F64_BYTES;

    // Ingestion: one owned copy of the batch, latency against a plain column copy.
    let (loaded, load_time) = best_of(3, || tabular_from_record_batch(&record_batch).unwrap());
    let (_, baseline_time) =
        best_of(3, || columns.iter().map(|(_, values)| values.clone()).collect::<Vec<Vec<f64>>>());
    let (_, load_peak) =
        counting::peak_above_start(|| tabular_from_record_batch(&record_batch).unwrap());
    assert_eq!(loaded.bytes_borrowed, 0, "the record-batch route owns its columns");
    assert!(loaded.bytes_copied >= input_bytes, "every column is copied once");
    assert!(loaded.bytes_copied <= 2 * input_bytes, "{} bytes copied", loaded.bytes_copied);
    assert!(
        load_peak <= declared("wide_adjustment", "load_peak_percent_of_input") * input_bytes / 100,
        "load peak {load_peak} over {input_bytes} input bytes"
    );
    assert!(
        load_time <= baseline_time * declared("wide_adjustment", "load_latency_factor") as u32,
        "load {load_time:?} against a plain copy {baseline_time:?}"
    );
    let data = loaded.data;

    // Design: the adjustment set plus treatment, over every row, borrows each column.
    let design_ids: Vec<VariableId> = ids(&data)[..covariates + 1].to_vec();
    let views: Vec<_> = design_ids.iter().map(|id| data.column(*id).unwrap()).collect();
    let all_rows: Vec<GlobalId> = (0..rows as u64).map(GlobalId::new).collect();
    let (full, full_peak) =
        counting::peak_above_start(|| DesignLayout::from_views(&views, &all_rows).unwrap());
    assert!(full.receipt().is_empty(), "an in-order contiguous run owns nothing");
    assert!(
        full_peak
            <= declared("wide_adjustment", "borrowed_layout_peak_bytes_per_column")
                * views.len() as u64,
        "borrowed layout peak {full_peak}"
    );
    for (index, (_, values)) in columns[..covariates + 1].iter().enumerate() {
        assert_eq!(full.column(index).unwrap(), values.as_slice());
    }

    // A strided cohort gathers one named copy per column and no more.
    let strided: Vec<GlobalId> = (0..rows as u64).step_by(2).map(GlobalId::new).collect();
    let (gathered, gather_peak) =
        counting::peak_above_start(|| DesignLayout::from_views(&views, &strided).unwrap());
    let copies = gathered.receipt().copies();
    assert_eq!(copies.len(), views.len());
    assert!(copies.iter().all(|copy| copy.name() == OwnershipCopy::GATHER));
    let gathered_bytes = (strided.len() * views.len()) as u64 * F64_BYTES;
    assert!(gather_peak >= gathered_bytes, "peak {gather_peak} is below the copies it names");
    assert!(
        gather_peak
            <= gathered_bytes
                + declared("wide_adjustment", "borrowed_layout_peak_bytes_per_column")
                    * views.len() as u64,
        "gather peak {gather_peak} holds more than the {gathered_bytes} bytes the receipt names"
    );
    assert_eq!(gathered.column(3).unwrap()[5], columns[3].1[10]);

    // Identity: the Arrow batch, the Parquet file and the Arrow IPC file are one table.
    let scratch = Scratch::new("wide");
    let names: Vec<String> = columns.iter().map(|(name, _)| name.clone()).collect();
    let parquet = scratch.0.join("wide.parquet");
    let ipc = scratch.0.join("wide.arrow");
    write_parquet(&parquet, &record_batch, 8192);
    write_ipc(&ipc, &record_batch);
    let digest = data.storage().content_digest();
    let from_parquet = read_file(&parquet, &names);
    let from_ipc = read_file(&ipc, &names);
    assert_eq!(from_parquet.storage().content_digest(), digest, "parquet route identity");
    assert_eq!(from_ipc.storage().content_digest(), digest, "ipc route identity");
    let mut changed = columns.clone();
    changed[covariates + 1].1[rows - 1] += 1.0;
    let other = tabular_from_record_batch(&batch(&changed)).unwrap().data;
    assert_ne!(other.storage().content_digest(), digest, "one changed value is a new identity");

    record("wide_adjustment", "input_bytes", input_bytes);
    record("wide_adjustment", "load_ns", load_time.as_nanos());
    record("wide_adjustment", "baseline_copy_ns", baseline_time.as_nanos());
    record("wide_adjustment", "load_peak_bytes", load_peak);
    record("wide_adjustment", "borrowed_layout_peak_bytes", full_peak);
    record("wide_adjustment", "gather_peak_bytes", gather_peak);
}

fn series_columns(rows: usize, seed: u64, variables: usize) -> Vec<(String, Vec<f64>)> {
    let mut next = lcg(seed);
    (0..variables).map(|v| (format!("v{v}"), (0..rows).map(|_| next()).collect())).collect()
}

fn series(columns: &[(String, Vec<f64>)]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())),
        1,
    )
    .unwrap()
}

#[test]
fn panel_and_environment_lag_windows_stay_inside_their_unit_with_bounded_allocation() {
    let units = declared("panel_environment_window", "units") as usize;
    let rows = declared("panel_environment_window", "rows_per_unit") as usize;
    let variables = declared("panel_environment_window", "variables") as usize;
    let max_lag = declared("panel_environment_window", "max_lag") as u32;
    let policy = KernelPolicy::default_policy();

    let sources: Vec<Vec<(String, Vec<f64>)>> =
        (0..units).map(|u| series_columns(rows, 100 + u as u64, variables)).collect();
    // Each unit arrives as its own Arrow batch and is prepared by the direct route.
    let panel: Vec<TimeSeriesData> = sources
        .iter()
        .map(|columns| {
            let loaded = tabular_from_record_batch(&batch(columns)).unwrap();
            TimeSeriesData::try_new(
                loaded.data.storage().clone(),
                antecedent_data::TimeIndex {
                    regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
                    length: rows,
                },
            )
            .unwrap()
        })
        .collect();
    let direct: Vec<TimeSeriesData> = sources.iter().map(|columns| series(columns)).collect();
    for (arrow, owned) in panel.iter().zip(&direct) {
        assert_eq!(arrow.storage().content_digest(), owned.storage().content_digest());
    }
    let variable_ids: Vec<VariableId> =
        panel[0].schema().variables().iter().map(|v| v.id).collect();

    // Pooled lag design: each unit borrows its source buffer, the stack owns one copy.
    let ((pooled, pooled_peak), pooled_time) = best_of(3, || {
        counting::peak_above_start(|| {
            LaggedFrame::from_panel(&panel, &variable_ids, max_lag, &policy).unwrap()
        })
    });
    let per_unit_rows = rows - max_lag as usize;
    assert_eq!(pooled.n_effective(), units * per_unit_rows);
    assert_eq!(pooled.ncols(), variables * (max_lag as usize + 1));
    let owned_bytes = pooled.values_bytes();
    assert!(pooled_peak >= owned_bytes, "peak {pooled_peak} is below the stacked copy");
    assert!(
        pooled_peak
            <= declared("panel_environment_window", "peak_percent_of_owned") * owned_bytes / 100,
        "pooled peak {pooled_peak} over the {owned_bytes} bytes the stack owns"
    );
    // No row takes a lagged parent from the previous unit's tail.
    let lag1 = pooled.column_index(variable_ids[0], antecedent_core::Lag::from_raw(1)).unwrap();
    let lag0 = pooled.column_index(variable_ids[0], antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
    for unit in 0..units {
        let start = unit * per_unit_rows;
        let source = &sources[unit][0].1;
        assert_eq!(pooled.column(lag0)[start], source[max_lag as usize]);
        assert_eq!(pooled.column(lag1)[start], source[max_lag as usize - 1]);
    }
    let again = LaggedFrame::from_panel(&direct, &variable_ids, max_lag, &policy).unwrap();
    for column in 0..pooled.ncols() {
        assert_eq!(pooled.column(column), again.column(column), "arrow and direct units agree");
    }

    // Environments: the same frame over a multi-environment list.
    let environments = MultiEnvironmentData::try_new(direct.clone()).unwrap();
    let env_frame =
        LaggedFrame::from_panel(environments.environments(), &variable_ids, max_lag, &policy)
            .unwrap();
    assert_eq!(env_frame.n_effective(), pooled.n_effective());

    // Identity: the Parquet route returns a unit with the Arrow route's digest.
    let scratch = Scratch::new("panel");
    let path = scratch.0.join("unit.parquet");
    write_parquet(&path, &batch(&sources[0]), 4096);
    let names: Vec<String> = sources[0].iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(
        read_file(&path, &names).storage().content_digest(),
        panel[0].storage().content_digest()
    );

    record("panel_environment_window", "owned_bytes", owned_bytes);
    record("panel_environment_window", "pooled_peak_bytes", pooled_peak);
    record("panel_environment_window", "pooled_ns", pooled_time.as_nanos());
}

#[test]
fn one_shared_design_serves_many_consumers_without_another_copy() {
    let rows = declared("shared_design", "rows") as usize;
    let columns_n = declared("shared_design", "columns") as usize;
    let consumers = declared("shared_design", "consumers") as usize;
    let columns = series_columns(rows, 7, columns_n);
    let data = tabular_from_record_batch(&batch(&columns)).unwrap().data;
    let views: Vec<_> = ids(&data).iter().map(|id| data.column(*id).unwrap()).collect();
    let all_rows: Vec<GlobalId> = (0..rows as u64).map(GlobalId::new).collect();

    let (layout, build_peak) =
        counting::peak_above_start(|| DesignLayout::from_views(&views, &all_rows).unwrap());
    assert!(layout.receipt().is_empty());

    // Each consumer reads its own column of the one layout. Nothing is allocated per consumer.
    let mut sums = vec![0.0_f64; consumers];
    let (_, consumer_peak) = counting::peak_above_start(|| {
        for (consumer, slot) in sums.iter_mut().enumerate() {
            let column = layout.column(consumer % columns_n).unwrap();
            *slot = column.iter().sum();
        }
    });
    assert_eq!(consumer_peak, 0, "a consumer of the shared layout allocates nothing");
    for (consumer, total) in sums.iter().enumerate() {
        let expected: f64 = columns[consumer % columns_n].1.iter().sum();
        assert_eq!(*total, expected);
    }
    // The layout is the table's own buffer for every consumer.
    for index in 0..columns_n {
        let owned = match &views[index] {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice().as_ptr(),
            _ => unreachable!("float columns"),
        };
        assert_eq!(layout.column(index).unwrap().as_ptr(), owned);
    }
    assert!(
        build_peak <= declared("shared_design", "build_peak_bytes_per_column") * columns_n as u64,
        "shared layout build peak {build_peak}"
    );

    // Identity: the layout reads the same bits through the Parquet route.
    let scratch = Scratch::new("shared");
    let path = scratch.0.join("shared.parquet");
    write_parquet(&path, &batch(&columns), 8192);
    let names: Vec<String> = columns.iter().map(|(name, _)| name.clone()).collect();
    assert_eq!(
        read_file(&path, &names).storage().content_digest(),
        data.storage().content_digest()
    );

    record("shared_design", "build_peak_bytes", build_peak);
    record("shared_design", "consumer_peak_bytes", consumer_peak);
}

#[test]
fn no_polars_crate_is_declared_or_locked_and_the_wheel_links_only_the_declared_features() {
    const LOCK: &str = include_str!("../../../Cargo.lock");
    const WORKSPACE: &str = include_str!("../../../Cargo.toml");
    const DATA: &str = include_str!("../Cargo.toml");
    const PYPROJECT: &str = include_str!("../../../python/pyproject.toml");
    const PY_CRATE: &str = include_str!("../../../python/Cargo.toml");
    for (name, text) in [
        ("Cargo.lock", LOCK),
        ("workspace Cargo.toml", WORKSPACE),
        ("antecedent-data Cargo.toml", DATA),
        ("pyproject.toml", PYPROJECT),
        ("python/Cargo.toml", PY_CRATE),
    ] {
        let linked = text.lines().any(|line| {
            let code = line.split('#').next().unwrap_or("").trim().to_ascii_lowercase();
            code.starts_with("name = \"polars")
                || code.starts_with("polars")
                || code.starts_with("\"polars")
                || code.contains("dep:polars")
        });
        assert!(!linked, "{name} names a Polars crate or requirement");
    }
    // The direct route's file readers are the only table backends the data crate offers.
    assert!(DATA.contains("parquet = [\"arrow\", \"dep:parquet\"]"));
    assert!(DATA.contains("ipc = [\"arrow\", \"dep:arrow-ipc\"]"));
    assert!(DATA.contains("default = [\"arrow\"]"));
    assert_eq!(declared("packaging", "polars_crates_locked"), 0);
    assert!(
        PYPROJECT.contains("\"numpy>=1.26\"") && PYPROJECT.contains("\"pydantic>="),
        "the wheel's runtime requirements are numpy and pydantic"
    );
}
