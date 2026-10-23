//! A Parquet table larger than the admission pool: materialization is refused
//! with its quote before reading, and the bounded weighted Gaussian capacity
//! route streams the same file to the summary a resident run returns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use std::path::PathBuf;

use antecedent::analysis::file_table::{
    CapacityControl, CapacityOptions, FileSourceError, FileTableError, GaussianSummarySpec,
    MemoryStage, ReadOptions, SummaryMissing, TableFile, TableFileFormat, WriteOptions,
    summarize_weighted_gaussian, write_table_file,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_core::execution_facts::OutputRetention;
use antecedent_data::TabularData;
use antecedent_io_core::capacity_account::{ResidentWeightedRows, run_weighted_capacity};

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
