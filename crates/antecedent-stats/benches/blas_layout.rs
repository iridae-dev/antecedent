//! Layout and crossover study for the dense products and factorizations the stats crate runs.
//!
//! Every configuration of an operation (reference loop or `faer`, 1, 4 and 16 workers, dense
//! column-major, dense row-major and column-sparse storage) is checked against an independent
//! reference before it is sampled, then timed. The output is a `[[case]]` record in the format
//! of `parity/blas_layout/layout.toml`:
//!
//! ```text
//! cargo bench -p antecedent-stats --bench blas_layout > /tmp/layout_cases.toml
//! python3 parity/blas_layout/derive_selections.py parity/blas_layout/layout.toml \
//!     --cases /tmp/layout_cases.toml --qr-crossover parity/solver_scope/crossover.toml \
//!     --write --write-rust crates/antecedent-stats/src/layout_measured.rs
//! ```
//!
//! `ANTECEDENT_LAYOUT_OPS` (comma separated: `gram`, `qr`, `svd`, `eigen`) restricts the
//! operations, `ANTECEDENT_LAYOUT_RECORD` also writes the record to a file, and
//! `ANTECEDENT_LAYOUT_QUICK=1` runs the small half of every shape list. Without `--bench` (for
//! example under `cargo test --benches`) the study runs one tiny shape per operation, once,
//! with the same accuracy checks.
//!
//! The QR case times the pivoted factorization of a fresh copy of the design; the copy is the
//! same for every worker count. Sixteen workers oversubscribe a host with fewer hardware
//! threads, which is recorded as measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, reason = "benchmark crate root")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "benchmark kernels index column-major buffers by explicit coordinates"
)]

use std::fmt::Write as _;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use antecedent_stats::layout_rules::{LayoutOp, scratch_requirement_bytes};
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
use faer::linalg::qr::col_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::svd::{ComputeSvdVectors, svd, svd_scratch};
use faer::{Accum, Mat, MatMut, MatRef, Par, Spec};

const WORKERS: [u32; 3] = [1, 4, 16];

/// Timing of one configuration, per call, in nanoseconds.
struct Timing {
    median: f64,
    low: f64,
    high: f64,
}

/// Median and quartiles of repeated timings; `smoke` takes one call.
fn sample(smoke: bool, mut run: impl FnMut()) -> Timing {
    run();
    if smoke {
        let start = Instant::now();
        run();
        let one = start.elapsed().as_nanos() as f64;
        return Timing { median: one, low: one, high: one };
    }
    run();
    let start = Instant::now();
    run();
    let one = (start.elapsed().as_nanos() as f64).max(1.0);
    let samples = if one > 100_000_000.0 { 5 } else { 15 };
    let iterations = ((3_000_000.0 / one) as u64).clamp(1, 100_000);
    let mut per_call: Vec<f64> = (0..samples)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                run();
            }
            start.elapsed().as_nanos() as f64 / iterations as f64
        })
        .collect();
    per_call.sort_by(f64::total_cmp);
    Timing {
        median: per_call[samples / 2],
        low: per_call[samples / 4],
        high: per_call[3 * samples / 4],
    }
}

fn par_of(workers: u32) -> Par {
    match NonZeroUsize::new(workers as usize) {
        Some(count) if workers > 1 => Par::Rayon(count),
        _ => Par::Seq,
    }
}

struct Record {
    text: String,
}

impl Record {
    #[allow(clippy::too_many_arguments, reason = "one flat record line per measured case")]
    fn case(
        &mut self,
        op: &str,
        layout: &str,
        rows: usize,
        cols: usize,
        nonzeros: Option<usize>,
        product: &str,
        workers: u32,
        scratch_bytes: u64,
        timing: &Timing,
    ) {
        let _ = write!(
            self.text,
            "[[case]]\nop = \"{op}\"\nlayout = \"{layout}\"\nrows = {rows}\ncols = {cols}\n"
        );
        if let Some(nonzeros) = nonzeros {
            let _ = writeln!(self.text, "nonzeros_per_column = {nonzeros}");
        }
        let _ = write!(
            self.text,
            "product = \"{product}\"\nworkers = {workers}\nmedian_ns = {}\nlow_ns = {}\nhigh_ns = \
             {}\nscratch_bytes = {}\n\n",
            timing.median.round() as u64,
            timing.low.round() as u64,
            timing.high.round() as u64,
            scratch_bytes.min(i64::MAX as u64),
        );
    }
}

fn design(n: usize, p: usize) -> Vec<f64> {
    (0..p)
        .flat_map(|col| {
            (0..n).map(move |row| ((row as f64 + 1.0) * (col as f64 + 1.3) * 0.17).sin())
        })
        .collect()
}

/// Column-major design with `nonzeros` pseudo-random nonzero rows per column.
fn sparse_design(n: usize, p: usize, nonzeros: usize) -> (Vec<f64>, Vec<Vec<(usize, f64)>>) {
    let mut dense = vec![0.0; n * p];
    let mut columns = Vec::with_capacity(p);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for col in 0..p {
        let mut rows: Vec<usize> = Vec::with_capacity(nonzeros);
        while rows.len() < nonzeros.min(n) {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let row = (state >> 33) as usize % n;
            if !rows.contains(&row) {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        let entries: Vec<(usize, f64)> = rows
            .into_iter()
            .map(|row| (row, ((row as f64 + 1.0) * (col as f64 + 1.3) * 0.17).sin() + 1.5))
            .collect();
        for &(row, value) in &entries {
            dense[col * n + row] = value;
        }
        columns.push(entries);
    }
    (dense, columns)
}

fn scalar_gram_colmajor(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for r in 0..n {
                sum += x[i * n + r] * x[j * n + r];
            }
            out[i * p + j] = sum;
            out[j * p + i] = sum;
        }
    }
}

fn scalar_gram_rowmajor(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    out[..p * p].fill(0.0);
    for r in 0..n {
        let row = &x[r * p..(r + 1) * p];
        for i in 0..p {
            let xi = row[i];
            for j in i..p {
                out[i * p + j] += xi * row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            out[i * p + j] = out[j * p + i];
        }
    }
}

fn sparse_dot(a: &[(usize, f64)], b: &[(usize, f64)]) -> f64 {
    let (mut i, mut j, mut sum) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

fn sparse_gram(columns: &[Vec<(usize, f64)>], out: &mut [f64]) {
    let p = columns.len();
    for i in 0..p {
        for j in i..p {
            let dot = sparse_dot(&columns[i], &columns[j]);
            out[i * p + j] = dot;
            out[j * p + i] = dot;
        }
    }
}

fn faer_gram_colmajor(x: &[f64], n: usize, p: usize, out: &mut [f64], par: Par) {
    let view = MatRef::from_column_major_slice(x, n, p);
    let target = MatMut::from_row_major_slice_mut(out, p, p);
    faer::linalg::matmul::matmul(target, Accum::Replace, view.transpose(), view, 1.0, par);
}

fn faer_gram_rowmajor(x: &[f64], n: usize, p: usize, out: &mut [f64], par: Par) {
    let view = MatRef::from_row_major_slice(x, n, p);
    let target = MatMut::from_row_major_slice_mut(out, p, p);
    faer::linalg::matmul::matmul(target, Accum::Replace, view.transpose(), view, 1.0, par);
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= 1e-8 * (1.0 + scale), "{what}: {a} vs {e}");
    }
}

fn study_gram(
    record: &mut Record,
    shapes: &[(usize, usize)],
    sparse: &[(usize, usize, usize)],
    smoke: bool,
) {
    for &(n, p) in shapes {
        let x = design(n, p);
        let mut expected = vec![0.0; p * p];
        scalar_gram_colmajor(&x, n, p, &mut expected);
        let mut out = vec![0.0; p * p];
        let timing =
            sample(smoke, || scalar_gram_colmajor(black_box(&x), n, p, black_box(&mut out)));
        record.case("gram", "dense", n, p, None, "reference", 1, 0, &timing);
        for workers in WORKERS {
            faer_gram_colmajor(&x, n, p, &mut out, par_of(workers));
            assert_close(&out, &expected, "faer dense gram");
            let timing = sample(smoke, || {
                faer_gram_colmajor(black_box(&x), n, p, black_box(&mut out), par_of(workers));
            });
            record.case("gram", "dense", n, p, None, "faer", workers, 0, &timing);
        }
        // Same design stored row-major.
        let mut rowmajor = vec![0.0; n * p];
        for c in 0..p {
            for r in 0..n {
                rowmajor[r * p + c] = x[c * n + r];
            }
        }
        scalar_gram_rowmajor(&rowmajor, n, p, &mut out);
        assert_close(&out, &expected, "row-major reference gram");
        let timing =
            sample(smoke, || scalar_gram_rowmajor(black_box(&rowmajor), n, p, black_box(&mut out)));
        record.case("gram", "dense_rowmajor", n, p, None, "reference", 1, 0, &timing);
        for workers in WORKERS {
            faer_gram_rowmajor(&rowmajor, n, p, &mut out, par_of(workers));
            assert_close(&out, &expected, "faer row-major gram");
            let timing = sample(smoke, || {
                faer_gram_rowmajor(
                    black_box(&rowmajor),
                    n,
                    p,
                    black_box(&mut out),
                    par_of(workers),
                );
            });
            record.case("gram", "dense_rowmajor", n, p, None, "faer", workers, 0, &timing);
        }
    }
    for &(n, p, nonzeros) in sparse {
        let (dense, columns) = sparse_design(n, p, nonzeros);
        let mut expected = vec![0.0; p * p];
        scalar_gram_colmajor(&dense, n, p, &mut expected);
        let mut out = vec![0.0; p * p];
        sparse_gram(&columns, &mut out);
        assert_close(&out, &expected, "column-wise sparse gram");
        let timing = sample(smoke, || sparse_gram(black_box(&columns), black_box(&mut out)));
        record.case("gram", "sparse", n, p, Some(nonzeros), "reference", 1, 0, &timing);
        for workers in WORKERS {
            faer_gram_colmajor(&dense, n, p, &mut out, par_of(workers));
            assert_close(&out, &expected, "faer gram on sparse data");
            let timing = sample(smoke, || {
                faer_gram_colmajor(black_box(&dense), n, p, black_box(&mut out), par_of(workers));
            });
            record.case("gram", "sparse", n, p, Some(nonzeros), "faer", workers, 0, &timing);
        }
    }
}

fn qr_diagonal(x: &[f64], n: usize, p: usize, workers: u32) -> Vec<f64> {
    let par = par_of(workers);
    let mut qr = Mat::<f64>::from_fn(n, p, |r, c| x[c * n + r]);
    let block = recommended_block_size::<f64>(n, p);
    let required = qr_in_place_scratch::<usize, f64>(n, p, block, par, Spec::default());
    let mut buffer = MemBuffer::new(required);
    let size = n.min(p);
    let mut q_coeff = Mat::<f64>::zeros(block, size);
    let (mut forward, mut inverse) = (vec![0usize; p], vec![0usize; p]);
    let _ = qr_in_place(
        qr.as_mut(),
        q_coeff.as_mut(),
        &mut forward,
        &mut inverse,
        par,
        MemStack::new(&mut buffer),
        Spec::default(),
    );
    (0..size).map(|i| qr[(i, i)].abs()).collect()
}

fn study_qr(record: &mut Record, shapes: &[(usize, usize)], smoke: bool) {
    for &(n, p) in shapes {
        let x = design(n, p);
        let expected = qr_diagonal(&x, n, p, 1);
        for workers in WORKERS {
            assert_close(&qr_diagonal(&x, n, p, workers), &expected, "pivoted QR diagonal");
            let par = par_of(workers);
            let block = recommended_block_size::<f64>(n, p);
            let required: StackReq =
                qr_in_place_scratch::<usize, f64>(n, p, block, par, Spec::default());
            let mut buffer = MemBuffer::new(required);
            let size = n.min(p);
            let mut q_coeff = Mat::<f64>::zeros(block, size);
            let (mut forward, mut inverse) = (vec![0usize; p], vec![0usize; p]);
            let timing = sample(smoke, || {
                let mut qr = Mat::<f64>::from_fn(n, p, |r, c| x[c * n + r]);
                let _ = qr_in_place(
                    qr.as_mut(),
                    q_coeff.as_mut(),
                    &mut forward,
                    &mut inverse,
                    par,
                    MemStack::new(&mut buffer),
                    Spec::default(),
                );
                black_box(&qr);
            });
            let bytes = scratch_requirement_bytes(LayoutOp::ColumnPivotedQr, n, p, workers);
            record.case("qr", "dense", n, p, None, "faer", workers, bytes, &timing);
        }
    }
}

fn singular_values(
    x: &[f64],
    n: usize,
    p: usize,
    workers: u32,
    timed: Option<&mut dyn FnMut(&mut dyn FnMut())>,
) -> Vec<f64> {
    let par = par_of(workers);
    let a = MatRef::from_column_major_slice(x, n, p);
    let compute = ComputeSvdVectors::Thin;
    let required = svd_scratch::<f64>(n, p, compute, compute, par, Spec::default());
    let mut buffer = MemBuffer::new(required);
    let size = n.min(p);
    let mut u = Mat::<f64>::zeros(n, size);
    let mut v = Mat::<f64>::zeros(p, size);
    let mut s = Diag::<f64>::zeros(size);
    let mut run = || {
        svd(
            a,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            par,
            MemStack::new(&mut buffer),
            Spec::default(),
        )
        .expect("the study designs converge");
    };
    match timed {
        Some(driver) => driver(&mut run),
        None => run(),
    }
    (0..size).map(|i| s.column_vector()[i]).collect()
}

fn study_svd(record: &mut Record, shapes: &[(usize, usize)], smoke: bool) {
    for &(n, p) in shapes {
        let x = design(n, p);
        let expected = singular_values(&x, n, p, 1, None);
        for workers in WORKERS {
            assert_close(&singular_values(&x, n, p, workers, None), &expected, "singular values");
            let mut timing = None;
            let mut driver = |run: &mut dyn FnMut()| timing = Some(sample(smoke, || run()));
            let _ = singular_values(&x, n, p, workers, Some(&mut driver));
            let bytes = scratch_requirement_bytes(LayoutOp::ThinSvd, n, p, workers);
            if let Some(timing) = timing {
                record.case("svd", "dense", n, p, None, "faer", workers, bytes, &timing);
            }
        }
    }
}

fn symmetric(n: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(n, n, |i, j| {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        ((lo as f64 + 1.0) * (hi as f64 + 1.7) * 0.013).sin() + if i == j { 2.0 } else { 0.0 }
    })
}

fn eigenvalues(
    matrix: &Mat<f64>,
    workers: u32,
    timed: Option<&mut dyn FnMut(&mut dyn FnMut())>,
) -> Vec<f64> {
    let n = matrix.nrows();
    let par = par_of(workers);
    let required =
        self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Spec::default());
    let mut buffer = MemBuffer::new(required);
    let mut values = Diag::<f64>::zeros(n);
    let mut vectors = Mat::<f64>::zeros(n, n);
    let mut run = || {
        self_adjoint_evd(
            matrix.as_ref(),
            values.as_mut(),
            Some(vectors.as_mut()),
            par,
            MemStack::new(&mut buffer),
            Spec::default(),
        )
        .expect("the study matrices converge");
    };
    match timed {
        Some(driver) => driver(&mut run),
        None => run(),
    }
    (0..n).map(|i| values.column_vector()[i]).collect()
}

fn study_eigen(record: &mut Record, orders: &[usize], smoke: bool) {
    for &n in orders {
        let matrix = symmetric(n);
        let expected = eigenvalues(&matrix, 1, None);
        for workers in WORKERS {
            assert_close(&eigenvalues(&matrix, workers, None), &expected, "eigenvalues");
            let mut timing = None;
            let mut driver = |run: &mut dyn FnMut()| timing = Some(sample(smoke, || run()));
            let _ = eigenvalues(&matrix, workers, Some(&mut driver));
            let bytes = scratch_requirement_bytes(LayoutOp::SymmetricEigen, n, n, workers);
            if let Some(timing) = timing {
                record.case("eigen", "dense", n, n, None, "faer", workers, bytes, &timing);
            }
        }
    }
}

fn wants(op: &str) -> bool {
    std::env::var("ANTECEDENT_LAYOUT_OPS")
        .map_or(true, |list| list.split(',').any(|item| item.trim() == op))
}

fn main() {
    let measuring = std::env::args().any(|arg| arg == "--bench");
    let smoke = !measuring;
    let quick = smoke || std::env::var("ANTECEDENT_LAYOUT_QUICK").is_ok_and(|v| v == "1");
    let mut record = Record { text: String::new() };
    let hardware = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let _ = writeln!(record.text, "# hardware threads available to this process: {hardware}\n");

    if wants("gram") {
        let (dense, sparse): (&[(usize, usize)], &[(usize, usize, usize)]) = if smoke {
            (&[(64, 8)], &[(128, 16, 4)])
        } else if quick {
            (&[(64, 8), (256, 16), (1024, 32)], &[(128, 16, 4), (1024, 64, 8)])
        } else {
            (
                &[(64, 8), (256, 16), (1024, 32), (4096, 128), (16384, 256)],
                &[(128, 16, 4), (1024, 64, 8), (8192, 1024, 48)],
            )
        };
        study_gram(&mut record, dense, sparse, smoke);
    }
    if wants("qr") {
        let shapes: &[(usize, usize)] = if smoke {
            &[(32, 8)]
        } else if quick {
            &[(32, 8), (64, 16), (128, 16), (256, 32), (512, 32)]
        } else {
            &[
                (32, 8),
                (64, 16),
                (128, 16),
                (256, 32),
                (512, 32),
                (1024, 64),
                (2048, 64),
                (8192, 128),
            ]
        };
        study_qr(&mut record, shapes, smoke);
    }
    if wants("svd") {
        let shapes: &[(usize, usize)] = if smoke {
            &[(64, 8)]
        } else if quick {
            &[(64, 8), (256, 32)]
        } else {
            &[(64, 8), (256, 32), (1024, 64), (4096, 128)]
        };
        study_svd(&mut record, shapes, smoke);
    }
    if wants("eigen") {
        let orders: &[usize] = if smoke {
            &[16]
        } else if quick {
            &[16, 64, 128]
        } else {
            &[16, 64, 256, 512]
        };
        study_eigen(&mut record, orders, smoke);
    }

    if measuring {
        print!("{}", record.text);
        if let Ok(path) = std::env::var("ANTECEDENT_LAYOUT_RECORD") {
            std::fs::write(path, &record.text).expect("write the layout record");
        }
    }
}
