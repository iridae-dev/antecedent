//! The measurement record reads what the statistics engine actually did.
//!
//! The workload builds a design matrix and solves least squares on it. The design builder
//! reports its buffer copies at the point it makes them, so the copied bytes in the record
//! must equal what the builder copied for these shapes, and must grow with the rows and
//! the covariates. The counting allocator must see at least the matrix the builder owns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test shapes are small literals; ids and indexes fit their types"
)]

use antecedent_core::measurement_record::{MeasurementRecord, MeasurementSummary, measure};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_kernels::native_probe::{CountingAllocator, NativeProbe};
use antecedent_stats::{CompiledDesign, FaerBackend, StatsError};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator::new();

/// Deterministic value in [0, 1) for slot `(i, k)` (splitmix64 finalizer).
fn unit(i: usize, k: usize) -> f64 {
    let mut x = (((i as u64) << 8) | k as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn columns(rows: usize, covariates: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let treatment: Vec<f64> = (0..rows).map(|i| f64::from(u8::from(unit(i, 99) < 0.4))).collect();
    let z: Vec<Vec<f64>> =
        (0..covariates).map(|k| (0..rows).map(|i| unit(i, k)).collect()).collect();
    let outcome: Vec<f64> = (0..rows)
        .map(|i| {
            let signal: f64 = z.iter().map(|column| column[i]).sum();
            2.0 * treatment[i] + 0.5 * signal + 0.1 * unit(i, 100)
        })
        .collect();
    (treatment, z, outcome)
}

fn run(rows: usize, covariates: usize) -> MeasurementRecord {
    let (treatment, z, outcome) = columns(rows, covariates);
    let refs: Vec<(VariableId, &[f64])> = z
        .iter()
        .enumerate()
        .map(|(k, c)| (VariableId::from_raw(k as u32 + 2), c.as_slice()))
        .collect();
    let (result, record) = measure(&ExecutionContext::for_tests(5), &NativeProbe, || {
        let design = CompiledDesign::linear_adjustment(&treatment, &refs, &outcome, &[])?;
        let coefficients = FaerBackend.least_squares_coefficients(
            &design.matrix,
            design.nrows,
            design.ncols,
            &design.outcome,
        )?;
        Ok::<_, StatsError>(coefficients)
    });
    let coefficients = result.unwrap();
    assert_eq!(coefficients.len(), 2 + covariates);
    record
}

/// Bytes `linear_adjustment` copies for these shapes: the three input columns plus the
/// covariates into the matrix, then the matrix, outcome and selection into shared storage.
fn designed_copy_bytes(rows: usize, covariates: usize) -> u64 {
    let columns = 2 + covariates;
    (rows * columns * 8 + rows * (columns + 3) * 8) as u64
}

#[test]
fn copied_and_allocated_bytes_follow_the_design_the_engine_built() {
    let small = run(400, 2);
    let more_rows = run(1600, 2);
    let more_columns = run(400, 8);

    assert_eq!(small.copy_bytes, designed_copy_bytes(400, 2));
    assert_eq!(more_rows.copy_bytes, designed_copy_bytes(1600, 2));
    assert_eq!(more_columns.copy_bytes, designed_copy_bytes(400, 8));
    assert_eq!(more_rows.copy_bytes, 4 * small.copy_bytes);
    assert!(more_columns.copy_bytes > small.copy_bytes);

    for (record, rows, covariates) in
        [(&small, 400, 2), (&more_rows, 1600, 2), (&more_columns, 400, 8)]
    {
        let matrix_bytes = (rows * (2 + covariates) * 8) as u64;
        assert!(record.observed("allocation_bytes"));
        assert!(
            record.allocation_bytes >= matrix_bytes,
            "allocated {} bytes for a {matrix_bytes}-byte matrix",
            record.allocation_bytes
        );
        assert!(record.peak_live_bytes >= matrix_bytes);
        assert!(!record.failed && !record.cancelled);
        // No model fit, test or spill boundary was crossed by this workload.
        assert!(!record.observed("fit_count"));
        assert!(!record.observed("spill_bytes"));
    }
    assert!(more_rows.allocation_bytes > small.allocation_bytes);

    let summary = MeasurementSummary::from_repeats(&[small.clone(), more_rows.clone()]);
    assert_eq!(summary.copy_bytes.min, small.copy_bytes);
    assert_eq!(summary.copy_bytes.max, more_rows.copy_bytes);

    a_refused_design_is_a_failed_record_with_the_error_intact();
}

/// Run inside the single test below: the allocator and its live-byte window are process-wide,
/// so concurrent windows in this binary would disturb each other's peaks.
fn a_refused_design_is_a_failed_record_with_the_error_intact() {
    let (treatment, _, outcome) = columns(50, 0);
    let short = &treatment[..10];
    let (result, record) = measure(&ExecutionContext::for_tests(5), &NativeProbe, || {
        CompiledDesign::linear_adjustment(short, &[], &outcome, &[])
    });
    assert!(result.is_err());
    assert!(record.failed);
    assert_eq!(record.copy_bytes, 0, "the refusal came before any buffer was copied");
}
