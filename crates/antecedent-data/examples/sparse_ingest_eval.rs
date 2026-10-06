//! One-off evaluation of dense versus CSC-style design-matrix materialization for a wide
//! binary-column adjustment set (the ~3,000-column case of TODO.md E7).
//!
//! Run with
//! `cargo run --release -p antecedent-data --example sparse_ingest_eval [-- ROWS COLS DENSITY]`
//! (defaults: 5000 rows, 3000 binary columns, density 0.02; a few seconds on a laptop).
//!
//! It times and counts allocation for the dense path the propensity preparer follows
//! (complete-case mask, one masked copy per column, a packed column-major design, an owned
//! `Arc` per column), against a CSC pattern built straight from the stored columns, then
//! the first use of each (`X'y`). The accepted schema and the estimand are untouched: both
//! paths read the same `TabularData` and must agree on `X'y` exactly.
//!
//! The CSC here is an evaluation device, not a second panel builder: nothing in the
//! library consumes it.

#![allow(clippy::float_cmp, reason = "binary columns are exactly 0.0 or 1.0 by contract")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use antecedent_data::{ColumnView, TableView, TabularData};

/// System allocator that tracks live, peak and cumulative bytes.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to `System`; the counters are bookkeeping only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Wall time and allocation of one measured step.
struct Cost {
    millis: f64,
    /// Peak live bytes above the level when the step started.
    peak_bytes: usize,
    /// Cumulative bytes requested during the step.
    total_bytes: usize,
}

/// Run `step` once, returning its value and cost.
fn measure<T>(step: impl FnOnce() -> T) -> (T, Cost) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let value = step();
    let millis = start.elapsed().as_secs_f64() * 1_000.0;
    let peak_bytes = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let total_bytes = TOTAL.load(Ordering::Relaxed);
    (value, Cost { millis, peak_bytes, total_bytes })
}

/// Run a step three times and return the last value with the median-time run's cost; earlier
/// values are dropped before the next run starts, so each run's peak is its own.
fn median_cost<T>(mut step: impl FnMut() -> T) -> (T, Cost) {
    let mut costs = Vec::with_capacity(3);
    let mut last = None;
    for _ in 0..3 {
        drop(last.take());
        let (value, cost) = measure(&mut step);
        costs.push(cost);
        last = Some(value);
    }
    costs.sort_by(|a, b| a.millis.total_cmp(&b.millis));
    (last.expect("three runs"), costs.swap_remove(1))
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn report(label: &str, cost: &Cost) {
    println!(
        "{label:<46} {:>9.1} ms {:>10.1} MiB peak {:>10.1} MiB allocated",
        cost.millis,
        mib(cost.peak_bytes),
        mib(cost.total_bytes)
    );
}

/// Deterministic uniform stream (64-bit LCG), enough for a synthetic 0/1 pattern.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// What the propensity preparer holds after building the design: the packed column-major
/// design (intercept first), each covariate as its own owned slice, the treatment and outcome.
struct DenseDesign {
    design: Arc<[f64]>,
    covariates: Vec<Arc<[f64]>>,
    outcome: Vec<f64>,
}

/// The dense path exactly as `propensity::prepare` follows it, over the public data API.
fn dense_design(
    data: &TabularData,
    adjustment: &[antecedent_core::VariableId],
    outcome: antecedent_core::VariableId,
    treatment: antecedent_core::VariableId,
) -> DenseDesign {
    let mut ids = vec![treatment, outcome];
    ids.extend_from_slice(adjustment);
    let mask = data.complete_case_mask(&ids).expect("complete-case mask");
    let t = data.float64_masked(treatment, &mask).expect("treatment");
    let y = data.float64_masked(outcome, &mask).expect("outcome");
    let nrows = t.len();
    let ncols = 1 + adjustment.len();
    let mut design = vec![0.0; nrows * ncols];
    design[..nrows].fill(1.0);
    let mut covariates = Vec::with_capacity(adjustment.len());
    for (i, &z) in adjustment.iter().enumerate() {
        let col = data.float64_masked(z, &mask).expect("covariate");
        let base = (1 + i) * nrows;
        design[base..base + nrows].copy_from_slice(&col);
        covariates.push(Arc::from(col));
    }
    DenseDesign { design: Arc::from(design), covariates, outcome: y }
}

/// Column-compressed pattern of an all-binary covariate block (a stored entry is a 1).
struct Csc {
    nrows: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<u32>,
}

/// CSC pattern read straight from the stored columns: one pass per column, no per-column
/// `f64` copy. A value other than exactly 0 or 1 ends the evaluation (the dense path is the
/// only one that carries non-binary columns).
fn csc_design(
    data: &TabularData,
    adjustment: &[antecedent_core::VariableId],
    outcome: antecedent_core::VariableId,
    treatment: antecedent_core::VariableId,
) -> Csc {
    let mut ids = vec![treatment, outcome];
    ids.extend_from_slice(adjustment);
    let mask = data.complete_case_mask(&ids).expect("complete-case mask");
    let nrows = mask.iter().filter(|k| **k).count();
    let mut col_ptr = Vec::with_capacity(adjustment.len() + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<u32> = Vec::new();
    for &z in adjustment {
        let ColumnView::Float64(column) = data.column(z).expect("covariate column") else {
            panic!("the evaluation covariates are float64 columns");
        };
        let mut kept = 0u32;
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                let v = column.values[i];
                assert!(v == 0.0 || v == 1.0, "non-binary value {v}: dense path only");
                if v == 1.0 {
                    row_idx.push(kept);
                }
                kept += 1;
            }
        }
        col_ptr.push(row_idx.len());
    }
    Csc { nrows, col_ptr, row_idx }
}

/// Scatter a CSC pattern into the packed dense design the dense path produces.
fn csc_to_dense(csc: &Csc) -> Vec<f64> {
    let ncols = csc.col_ptr.len();
    let mut design = vec![0.0; csc.nrows * ncols];
    design[..csc.nrows].fill(1.0);
    for j in 0..csc.col_ptr.len() - 1 {
        let base = (1 + j) * csc.nrows;
        for &r in &csc.row_idx[csc.col_ptr[j]..csc.col_ptr[j + 1]] {
            design[base + r as usize] = 1.0;
        }
    }
    design
}

/// `X'y` over the covariate columns of the dense design.
fn xty_dense(dense: &DenseDesign) -> Vec<f64> {
    let n = dense.outcome.len();
    (0..dense.covariates.len())
        .map(|j| {
            dense.design[(1 + j) * n..(2 + j) * n]
                .iter()
                .zip(&dense.outcome)
                .map(|(x, y)| x * y)
                .sum()
        })
        .collect()
}

/// `X'y` over the covariate columns of the CSC pattern.
fn xty_csc(csc: &Csc, outcome: &[f64]) -> Vec<f64> {
    (0..csc.col_ptr.len() - 1)
        .map(|j| {
            csc.row_idx[csc.col_ptr[j]..csc.col_ptr[j + 1]]
                .iter()
                .map(|&r| outcome[r as usize])
                .sum()
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().map_or(5_000, |a| a.parse().expect("ROWS"));
    let cols: usize = args.next().map_or(3_000, |a| a.parse().expect("COLS"));
    let density: f64 = args.next().map_or(0.02, |a| a.parse().expect("DENSITY"));
    println!("rows={rows} binary_columns={cols} target_density={density}\n");

    let before = LIVE.load(Ordering::Relaxed);
    let mut rng = Lcg(0x5DEE_CE66_D1CE_4E5B);
    let columns: Vec<Vec<f64>> = (0..cols)
        .map(|_| (0..rows).map(|_| f64::from(u8::from(rng.uniform() < density))).collect())
        .collect();
    let treatment: Vec<f64> = (0..rows).map(|_| f64::from(u8::from(rng.uniform() < 0.5))).collect();
    let outcome: Vec<f64> =
        (0..rows).map(|i| treatment[i] + columns[0][i] + rng.uniform()).collect();
    let mut named: Vec<(String, &[f64])> =
        columns.iter().enumerate().map(|(j, c)| (format!("x{j}"), c.as_slice())).collect();
    named.push(("t".into(), treatment.as_slice()));
    named.push(("y".into(), outcome.as_slice()));
    let data = TabularData::from_f64_columns(named).expect("dataset");
    drop(columns);
    let resident = LIVE.load(Ordering::Relaxed).saturating_sub(before);
    println!(
        "resident TabularData (accepted dense schema, unchanged): {:.1} MiB ({} f64 cells)\n",
        mib(resident),
        rows * (cols + 2)
    );

    let schema = data.schema();
    let adjustment: Vec<_> =
        (0..cols).map(|j| schema.id_of(&format!("x{j}")).expect("column")).collect();
    let t = schema.id_of("t").expect("t");
    let y = schema.id_of("y").expect("y");

    let (dense, dense_build) = median_cost(|| dense_design(&data, &adjustment, y, t));
    let (csc, csc_build) = median_cost(|| csc_design(&data, &adjustment, y, t));
    let nnz = csc.row_idx.len();
    let held_dense = (dense.design.len() + dense.covariates.iter().map(|c| c.len()).sum::<usize>())
        * std::mem::size_of::<f64>();
    let held_csc =
        nnz * std::mem::size_of::<u32>() + csc.col_ptr.len() * std::mem::size_of::<usize>();
    println!(
        "nnz={nnz} measured_density={:.4}; retained after build: dense {:.1} MiB, CSC {:.3} MiB\n",
        nnz as f64 / (rows * cols) as f64,
        mib(held_dense),
        mib(held_csc)
    );
    report("dense design build (current path)", &dense_build);
    report("CSC pattern build from stored columns", &csc_build);

    let (back, back_cost) = median_cost(|| csc_to_dense(&csc));
    assert_eq!(&back[..], &dense.design[..], "CSC scatter must reproduce the dense design");
    report("CSC -> dense materialization (dense consumer)", &back_cost);
    drop(back);

    let (dense_xty, dense_use) = median_cost(|| xty_dense(&dense));
    let (csc_xty, csc_use) = median_cost(|| xty_csc(&csc, &dense.outcome));
    assert_eq!(dense_xty, csc_xty, "X'y must agree exactly between the two representations");
    println!();
    report("first use X'y, dense", &dense_use);
    report("first use X'y, CSC", &csc_use);
    black_box((&dense_xty, &csc_xty));
    println!(
        "\nDense X'y == CSC X'y exactly ({} columns). Build speed-up CSC vs dense: {:.1}x; \
         retained-memory ratio dense/CSC: {:.0}x.",
        dense_xty.len(),
        dense_build.millis / csc_build.millis,
        held_dense as f64 / held_csc as f64
    );
}
