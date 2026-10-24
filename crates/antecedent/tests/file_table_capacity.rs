//! A Parquet table larger than the admission pool: materialization is refused
//! with its quote before reading, and the bounded weighted Gaussian capacity
//! route streams the same file to the summary a resident run returns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "the streamed summary must equal the resident result bit for bit, so the comparison is exact"
)]

use std::path::PathBuf;

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

use antecedent::analysis::file_table::{
    CapacityControl, CapacityOptions, ClosureError, ClosureInputs, ColumnFilter, ColumnRole,
    FileSourceError, FileTableError, GaussianSummarySpec, MemoryStage, ObservationFrame,
    PushdownCheckError, ReadOptions, RowFault, SummaryMissing, TableFile, TableFileFormat,
    WriteOptions, checked_projection, checked_push_filter, summarize_weighted_gaussian,
    summarize_weighted_gaussian_faulted, write_table_file,
};
use antecedent_contract::rewrite_legality::PreservationAxis;
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::physical_choice::PhysicalAlternative;
use antecedent_data::TabularData;
use antecedent_io_core::capacity_account::{
    ResidentWeightedRows, SpillOptions, WeightedCapacityError, WeightedCapacityOutput,
    WeightedCapacityResult, read_spilled_summary, run_weighted_capacity,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let dir = std::env::temp_dir()
            .join(format!("antecedent-file-table-{}-{label}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const ROWS: usize = 4_000;

fn columns(missing_row: Option<usize>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let x1: Vec<f64> = (0..ROWS).map(|i| ((i * 37) % 101) as f64 / 10.0).collect();
    let x2: Vec<f64> = (0..ROWS).map(|i| ((i * 13) % 7) as f64).collect();
    let w: Vec<f64> = (0..ROWS).map(|i| 0.5 + ((i * 11) % 5) as f64 / 4.0).collect();
    let mut y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + (((i * 7919) % 97) as f64 - 48.0) / 100.0)
        .collect();
    if let Some(row) = missing_row {
        y[row] = f64::NAN;
    }
    (x1, x2, w, y)
}

fn write(path: &std::path::Path, missing_row: Option<usize>) {
    let (x1, x2, w, y) = columns(missing_row);
    let data = TabularData::from_f64_columns([
        ("x1", &x1[..]),
        ("x2", &x2[..]),
        ("w", &w[..]),
        ("y", &y[..]),
        ("unused", &vec![0.0; ROWS][..]),
    ])
    .unwrap();
    write_table_file(path, TableFileFormat::Parquet, &data, WriteOptions { rows_per_group: 512 })
        .unwrap();
}

fn resident_design(skip: Option<usize>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (x1, x2, w, y) = columns(None);
    let mut design = Vec::new();
    let mut response = Vec::new();
    let mut weights = Vec::new();
    for row in 0..ROWS {
        if Some(row) == skip {
            continue;
        }
        design.extend_from_slice(&[1.0, x1[row], x2[row]]);
        response.push(y[row]);
        weights.push(w[row]);
    }
    (design, response, weights)
}

fn spec() -> GaussianSummarySpec {
    let mut spec = GaussianSummarySpec::new(vec!["x1".into(), "x2".into()], "y");
    spec.weight = Some("w".into());
    spec
}

#[test]
fn a_table_above_the_pool_is_refused_resident_and_streamed_as_a_summary() {
    let scratch = Scratch::new("stream");
    let path = scratch.0.join("rows.parquet");
    write(&path, None);
    let table = TableFile::open(&path).unwrap();
    let options = CapacityOptions::new(256, 1 << 20, OutputRetention::SummaryOnly);

    let plan =
        table.plan_columns(&["x1", "x2", "w", "y"], ReadOptions::with_chunk_rows(256)).unwrap();
    let quote = table.quote(&plan).unwrap().unwrap();
    let chunk = plan.chunk_quote().unwrap().total_bytes();
    let budget = 4 * chunk + 65_536;
    assert!(budget < quote.total_bytes(), "the pool holds a few chunks, not the table");
    let pool = AdmissionPool::fixed(1, budget);
    let study = ExecutionContext::for_tests(1).with_admission(pool.clone());

    let refused = table.materialize(&plan, &study).unwrap_err();
    let FileSourceError::MemoryRefused { stage, required_bytes, available_bytes, .. } = &refused
    else {
        panic!("the resident route is refused before reading: {refused}")
    };
    assert_eq!(*stage, MemoryStage::Materialization);
    assert_eq!(*required_bytes, quote.total_bytes());
    assert_eq!(*available_bytes, budget);
    assert!(refused.next_attempt().unwrap().contains("summary route"));
    assert_eq!(pool.available().bytes(), budget);

    let summary =
        summarize_weighted_gaussian(&table, &spec(), CapacityControl::Glm, &options, &study)
            .unwrap();
    assert_eq!(pool.available().bytes(), budget, "every chunk lease was returned");
    assert_eq!(summary.rows_read, ROWS as u64);
    assert_eq!(summary.rows_used, ROWS as u64);
    assert_eq!(summary.design_columns, vec!["(intercept)", "x1", "x2"]);
    assert_eq!(summary.file_identity.rows().get(), ROWS as u64);
    // The run reports the projection and scan it executed as physical operators: the
    // closure's columns, the 256-row chunk quote, and the Gram of the three-column design.
    let effects: Vec<_> = summary.scan.iter().map(|operator| operator.effect()).collect();
    assert_eq!(
        effects,
        [antecedent_core::PhysicalEffect::Select, antecedent_core::PhysicalEffect::Scan]
    );
    let scan = &summary.scan[1];
    assert!(scan.estimates().scratch().unwrap() > 0);
    assert_eq!(scan.estimates().retained(), Some((9 + 6 + 2) * 8));
    assert!(scan.cancellation_points().contains(antecedent_core::CancellationPoint::Scan));
    let streamed = summary.output.into_resident().unwrap();

    let (design, response, weights) = resident_design(None);
    let mut resident = ResidentWeightedRows::try_new(&design, &response, &weights).unwrap();
    let reference = run_weighted_capacity(CapacityControl::Glm, &mut resident, &options)
        .unwrap()
        .into_resident()
        .unwrap();
    assert_eq!(streamed.coefficients, reference.coefficients, "the same rows, the same summary");
    assert_eq!(streamed.content_identity, reference.content_identity);
    assert_eq!(streamed.cross_products, reference.cross_products);
    assert!((streamed.coefficients[1] - 2.0).abs() < 0.05);
}

#[test]
fn missing_values_are_refused_or_left_out_as_declared() {
    let scratch = Scratch::new("missing");
    let path = scratch.0.join("rows.parquet");
    write(&path, Some(123));
    let table = TableFile::open(&path).unwrap();
    let options = CapacityOptions::new(500, 1 << 20, OutputRetention::SummaryOnly);
    let study = ExecutionContext::for_tests(1);

    let refused =
        summarize_weighted_gaussian(&table, &spec(), CapacityControl::Glm, &options, &study)
            .unwrap_err();
    let FileTableError::Source(FileSourceError::MissingValue { column, row, .. }) = &refused else {
        panic!("a missing response is refused by default: {refused}")
    };
    assert_eq!(column, "y");
    assert_eq!(row.get(), 123);

    let mut complete = spec();
    complete.missing = SummaryMissing::CompleteCase;
    let summary =
        summarize_weighted_gaussian(&table, &complete, CapacityControl::Glm, &options, &study)
            .unwrap();
    assert_eq!(summary.rows_dropped, 1);
    assert_eq!(summary.rows_used, ROWS as u64 - 1);
    let streamed = summary.output.into_resident().unwrap();
    let (design, response, weights) = resident_design(Some(123));
    let mut resident = ResidentWeightedRows::try_new(&design, &response, &weights).unwrap();
    let reference = run_weighted_capacity(CapacityControl::Glm, &mut resident, &options)
        .unwrap()
        .into_resident()
        .unwrap();
    assert_eq!(streamed.coefficients, reference.coefficients);

    let absent = GaussianSummarySpec::new(vec!["nope".into()], "y");
    let refused =
        summarize_weighted_gaussian(&table, &absent, CapacityControl::Glm, &options, &study)
            .unwrap_err();
    assert!(matches!(refused, FileTableError::Source(FileSourceError::UnknownColumn { .. })));
}

#[test]
fn a_lowering_that_drops_repeats_or_alters_a_row_is_refused_at_the_file_summary() {
    let scratch = Scratch::new("faulted");
    let path = scratch.0.join("rows.parquet");
    write(&path, None);
    let table = TableFile::open(&path).unwrap();
    let options = CapacityOptions::new(256, 1 << 20, OutputRetention::SummaryOnly);
    let study = ExecutionContext::for_tests(1);

    let honest =
        summarize_weighted_gaussian(&table, &spec(), CapacityControl::Glm, &options, &study)
            .unwrap();
    assert_eq!(honest.rows_used, ROWS as u64);

    for fault in [RowFault::DropRow(10), RowFault::DuplicateRow(10), RowFault::AlterResponse(10)] {
        let refused = summarize_weighted_gaussian_faulted(
            &table,
            &spec(),
            CapacityControl::Glm,
            &options,
            &study,
            fault,
        )
        .unwrap_err();
        let FileTableError::Rewrite(refusal) = &refused else {
            panic!("{fault:?} must be refused as a rewrite: {refused}")
        };
        assert_eq!(refusal.axis(), PreservationAxis::SampleSelection, "{fault:?}");
        assert_eq!(refusal.kind(), Some("sufficient_statistics"), "{fault:?}");
    }
}

fn observation_frame() -> ObservationFrame {
    ObservationFrame::try_new(vec![
        (ColumnRole::new("a"), vec![true, true, true, true]),
        (ColumnRole::new("b"), vec![true, false, true, true]),
        (ColumnRole::new("w"), vec![true, true, false, true]),
    ])
    .unwrap()
}

#[test]
fn pushdowns_are_checked_against_the_rows_they_deliver() {
    let closure = ClosureInputs::new([ColumnRole::new("a")]).weight(ColumnRole::new("w")).close();
    let frame = observation_frame();

    let harmless = ColumnFilter::new(ColumnRole::new("b"), vec![false, false, true, false]);
    let cohort = checked_push_filter(&closure, &frame, &harmless).unwrap();
    assert_eq!(cohort.rows(), &[0, 1, 3]);

    let cuts_a_kept_row = ColumnFilter::new(ColumnRole::new("b"), vec![false, true, false, false]);
    let refused = checked_push_filter(&closure, &frame, &cuts_a_kept_row).unwrap_err();
    assert!(matches!(refused, PushdownCheckError::Closure(ClosureError::Refused(_))));

    let keep = [ColumnRole::new("a"), ColumnRole::new("w")];
    assert_eq!(checked_projection(&closure, &frame, &keep).unwrap().rows(), &[0, 1, 3]);

    let drops_required = [ColumnRole::new("a")];
    let refused = checked_projection(&closure, &frame, &drops_required).unwrap_err();
    assert!(matches!(refused, PushdownCheckError::Closure(ClosureError::Refused(_))));

    // Keeping the extra column is not refused by the closure, but completing cases on the
    // projected frame then loses row 1, which the declared cohort keeps.
    let wider = [ColumnRole::new("a"), ColumnRole::new("w"), ColumnRole::new("b")];
    let refused = checked_projection(&closure, &frame, &wider).unwrap_err();
    let PushdownCheckError::Rewrite(refusal) = &refused else {
        panic!("a wider projection changes the cohort: {refused}")
    };
    assert_eq!(refusal.axis(), PreservationAxis::SampleSelection);
    assert_eq!(refusal.kind(), Some("column_filter_pushdown"));
}

/// The summary of a run in either retention: resident as returned, or read back from its spill.
fn summary_of(output: WeightedCapacityOutput, dir: &std::path::Path) -> WeightedCapacityResult {
    match output {
        WeightedCapacityOutput::Resident(result) => result,
        WeightedCapacityOutput::Spilled(spilled) => {
            assert_eq!(spilled.retention, OutputRetention::DiskBacked);
            assert_eq!(spilled.charges.spill_bytes, spilled.handle.frame_bytes());
            let on_disk = std::fs::metadata(spilled.handle.path()).unwrap().len();
            assert_eq!(on_disk, spilled.charges.spill_bytes, "the charged frame is the file");
            assert!(spilled.handle.path().starts_with(dir));
            let back = read_spilled_summary(&spilled.handle).unwrap();
            assert_eq!(back.charges, spilled.charges);
            spilled.handle.remove().unwrap();
            back
        }
        _ => panic!("an unexpected capacity output form"),
    }
}

fn options_for(
    retention: OutputRetention,
    batch_rows: usize,
    ceiling_bytes: u64,
    spill: &std::path::Path,
) -> CapacityOptions {
    let mut options = CapacityOptions::new(batch_rows, ceiling_bytes, retention);
    if retention == OutputRetention::DiskBacked {
        options.spill = Some(SpillOptions { path: spill.to_path_buf(), pool: None });
    }
    options
}

/// Everything a summary reports except the control that was asked for.
fn assert_same_summary(left: &WeightedCapacityResult, right: &WeightedCapacityResult) {
    assert_eq!(left.retention, right.retention);
    assert_eq!(left.coefficients, right.coefficients);
    assert_eq!(left.robust_covariance, right.robust_covariance);
    assert_eq!(left.leverage, right.leverage);
    assert_eq!(left.residual, right.residual);
    assert_eq!(left.cross_products, right.cross_products);
    assert_eq!(left.fit, right.fit);
    assert_eq!(left.content_identity, right.content_identity);
    assert_eq!(left.charges, right.charges);
    assert_eq!(left.batches, right.batches);
    assert_eq!(left.source_passes, right.source_passes);
    assert_eq!(left.physical_choice().selected(), right.physical_choice().selected());
    assert_eq!(left.physical_choice().rejected(), right.physical_choice().rejected());
}

#[test]
fn both_controls_agree_through_the_file_route_in_both_retentions() {
    let scratch = Scratch::new("controls");
    let path = scratch.0.join("rows.parquet");
    write(&path, None);
    let table = TableFile::open(&path).unwrap();
    let plan =
        table.plan_columns(&["x1", "x2", "w", "y"], ReadOptions::with_chunk_rows(256)).unwrap();
    let chunk = plan.chunk_quote().unwrap().total_bytes();
    let budget = 4 * chunk + 65_536;

    let (design, response, weights) = resident_design(None);
    for retention in [OutputRetention::SummaryOnly, OutputRetention::DiskBacked] {
        let run = |control: CapacityControl| {
            let pool = AdmissionPool::fixed(1, budget);
            let study = ExecutionContext::for_tests(1).with_admission(pool.clone());
            // The output charge includes the spill handle's path bytes, so the two
            // controls spill to paths of equal length to compare their charges.
            let tag = if control == CapacityControl::Glm { "a" } else { "b" };
            let spill = scratch.0.join(format!("control-{tag}.spill"));
            let options = options_for(retention, 256, 1 << 22, &spill);
            let summary =
                summarize_weighted_gaussian(&table, &spec(), control, &options, &study).unwrap();
            assert_eq!(pool.available().bytes(), budget, "{control:?}: chunk leases returned");
            summary
        };
        let glm = run(CapacityControl::Glm);
        let bayes = run(CapacityControl::GaussianBayes);
        assert_eq!(glm.rows_read, bayes.rows_read);
        assert_eq!(glm.rows_used, bayes.rows_used);
        assert_eq!(glm.rows_dropped, bayes.rows_dropped);
        assert_eq!(glm.design_columns, bayes.design_columns);
        assert_eq!(glm.file_identity, bayes.file_identity);
        // The ceiling covers file decoding, ownership, identity, spill and output.
        let (glm_charges, bayes_charges) = (glm.output.charges(), bayes.output.charges());
        assert_eq!(glm_charges, bayes_charges);
        assert!(glm_charges.total_bytes <= 1 << 22);
        assert!(glm_charges.decoder_bytes >= chunk, "decoding is charged: {glm_charges:?}");
        assert_eq!(glm_charges.input_ownership_bytes, 0, "the file body is not owned");
        assert_eq!(glm_charges.spill_bytes > 0, retention == OutputRetention::DiskBacked);

        let glm = summary_of(glm.output, &scratch.0);
        let bayes = summary_of(bayes.output, &scratch.0);
        assert_eq!(glm.control, CapacityControl::Glm);
        assert_eq!(bayes.control, CapacityControl::GaussianBayes);
        assert_same_summary(&glm, &bayes);

        // Both equal the resident run of the same rows.
        let mut resident = ResidentWeightedRows::try_new(&design, &response, &weights).unwrap();
        let reference = run_weighted_capacity(
            CapacityControl::GaussianBayes,
            &mut resident,
            &CapacityOptions::new(256, 1 << 22, OutputRetention::SummaryOnly),
        )
        .unwrap()
        .into_resident()
        .unwrap();
        assert_eq!(glm.coefficients, reference.coefficients);
        assert_eq!(glm.cross_products, reference.cross_products);
        assert_eq!(bayes.content_identity, reference.content_identity);
    }
    let leftovers = std::fs::read_dir(&scratch.0).unwrap().count();
    assert_eq!(leftovers, 1, "only the table file remains; every spill was removed");
}

#[test]
fn measured_peak_with_decoding_ownership_spill_and_output_stays_inside_the_ceiling() {
    const BIG: usize = 120_000;
    const CHUNK: usize = 1024;
    let scratch = Scratch::new("measured");
    let path = scratch.0.join("big.parquet");
    {
        let x1: Vec<f64> = (0..BIG).map(|i| ((i * 37) % 101) as f64 / 10.0).collect();
        let x2: Vec<f64> = (0..BIG).map(|i| ((i * 13) % 7) as f64).collect();
        let w: Vec<f64> = (0..BIG).map(|i| 0.5 + ((i * 11) % 5) as f64 / 4.0).collect();
        let y: Vec<f64> = (0..BIG)
            .map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + ((i * 7919) % 97) as f64 / 100.0)
            .collect();
        let data = TabularData::from_f64_columns([
            ("x1", &x1[..]),
            ("x2", &x2[..]),
            ("w", &w[..]),
            ("y", &y[..]),
        ])
        .unwrap();
        write_table_file(
            &path,
            TableFileFormat::Parquet,
            &data,
            WriteOptions { rows_per_group: CHUNK },
        )
        .unwrap();
    }
    let table = TableFile::open(&path).unwrap();
    let table_column_bytes = (BIG * 4 * std::mem::size_of::<f64>()) as u64;
    let plan =
        table.plan_columns(&["x1", "x2", "w", "y"], ReadOptions::with_chunk_rows(CHUNK)).unwrap();
    let chunk = plan.chunk_quote().unwrap().total_bytes();
    let budget = 4 * chunk + 65_536;
    let spill = scratch.0.join("measured.spill");
    let run = |control: CapacityControl, ceiling: u64| {
        let pool = AdmissionPool::fixed(1, budget);
        let study = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let options = options_for(OutputRetention::DiskBacked, CHUNK, ceiling, &spill);
        let outcome = counting::peak_above_start(|| {
            summarize_weighted_gaussian(&table, &spec(), control, &options, &study)
        });
        assert_eq!(pool.available().bytes(), budget, "chunk leases returned");
        outcome
    };

    // The reservation, from an unbounded probe.
    let (probe, _) = run(CapacityControl::GaussianBayes, u64::MAX);
    let reserved = probe.unwrap().output.charges();
    summary_of(run(CapacityControl::GaussianBayes, u64::MAX).0.unwrap().output, &scratch.0);
    assert!(
        reserved.total_bytes < table_column_bytes,
        "the ceiling is below the table: {reserved:?}"
    );
    assert!(reserved.spill_bytes > 0 && reserved.decoder_bytes >= chunk);

    for control in [CapacityControl::Glm, CapacityControl::GaussianBayes] {
        let (outcome, peak) = run(control, reserved.total_bytes);
        let summary = outcome.unwrap();
        assert_eq!(summary.rows_used, BIG as u64);
        let charges = summary.output.charges();
        assert_eq!(charges, reserved, "{control:?} reserves the same bytes");
        assert!(charges.total_bytes <= reserved.total_bytes);
        assert!(peak > 0);
        assert!(
            peak <= charges.total_bytes,
            "{control:?}: measured peak {peak} exceeds the reservation {charges:?}"
        );
        assert!(peak < table_column_bytes, "{control:?}: peak {peak} holds the table");
        let result = summary_of(summary.output, &scratch.0);
        // The route is promised as a streamed reduction, and its bound is the measured peak
        // above, not a receipt claim.
        assert_eq!(result.physical_choice().selected(), PhysicalAlternative::StreamedReduction);
    }

    // One byte under the reservation is refused, naming the overage.
    let (refused, _) = run(CapacityControl::Glm, reserved.total_bytes - 1);
    let FileTableError::Capacity(WeightedCapacityError::OverCeiling { over_by, .. }) =
        refused.unwrap_err()
    else {
        panic!("one byte under the reservation is a ceiling refusal")
    };
    assert_eq!(over_by, 1);
    assert!(!spill.exists(), "a refused run leaves no spill file");
}
