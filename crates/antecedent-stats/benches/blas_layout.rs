//! Faer layouts versus the column-major reference.
//!
//! Dense Grams compare `form_xtx` on a column-major design with faer matmul. Sparse Grams compare
//! a column-wise reference product with the sparse-sparse product faer already provides. Shapes are
//! small and large. Worker counts are 1, 4, and 16, taken from `FaerScope`. One worker is the
//! serial path, and the small problems always take that path. Serial and parallel solutions are
//! checked before sampling. Measured medians and the selected path are in
//! `parity/blas_layout/layout.toml`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_stats::faer_scope::FaerScope;
use antecedent_stats::form_xtx;
use antecedent_stats::gram::ColumnMajorDesign;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};
use faer::sparse::{SparseColMat, SparseColMatMut, SparseColMatRef, SymbolicSparseColMat, Triplet};
use faer::{Accum, Mat, Par};

const SMALL_ROWS: usize = 64;
const SMALL_COLS: usize = 8;
const LARGE_ROWS: usize = 4_096;
const LARGE_COLS: usize = 128;
const SMALL_SPARSE_ROWS: usize = 128;
const SMALL_SPARSE_COLS: usize = 16;
const SMALL_NONZEROS_PER_COLUMN: usize = 4;
const LARGE_SPARSE_ROWS: usize = 8_192;
const LARGE_SPARSE_COLS: usize = 1_024;
const LARGE_NONZEROS_PER_COLUMN: usize = 48;
const SOLUTION_TOLERANCE: f64 = 1e-8;

struct DenseProblem {
    rows: usize,
    cols: usize,
    design: Vec<f64>,
    matrix: Mat<f64>,
}

struct ReferenceColumn {
    rows: Vec<usize>,
    values: Vec<f64>,
}

struct SparseFactors {
    matrix: SparseColMat<usize, f64>,
    transpose: SparseColMat<usize, f64>,
    symbolic: SymbolicSparseColMat<usize>,
    info: SparseMatMulInfo,
}

struct SparseProblem {
    cols: usize,
    columns: Vec<ReferenceColumn>,
    transpose_columns: Vec<ReferenceColumn>,
    factors: SparseFactors,
}

fn faer_scope(workers: u32) -> FaerScope {
    let scope = FaerScope::try_new(workers, u64::MAX).expect("at least one worker");
    let degree = scope.parallelism();
    if workers == 1 {
        assert!(matches!(degree, Par::Seq), "one worker must stay on the serial path");
    } else {
        let expected = usize::try_from(workers).expect("worker count");
        assert!(
            matches!(degree, Par::Rayon(count) if count.get() == expected),
            "wider scope must use the declared worker count",
        );
    }
    scope
}

fn serial_scope() -> FaerScope {
    faer_scope(1)
}

fn dense_design(rows: usize, cols: usize) -> Vec<f64> {
    let mut design = vec![0.0; rows * cols];
    for row in 0..rows {
        design[row] = 1.0;
        for col in 1..cols {
            let phase = (col as f64 + 1.3) * 0.17;
            design[col * rows + row] = ((row as f64 + 1.0) * phase).sin();
        }
    }
    design
}

fn dense_problem(rows: usize, cols: usize) -> DenseProblem {
    let design = dense_design(rows, cols);
    let matrix = Mat::from_fn(rows, cols, |row, col| design[col * rows + row]);
    DenseProblem { rows, cols, design, matrix }
}

fn fill_reference_dense(problem: &DenseProblem, gram: &mut [f64]) {
    let layout = ColumnMajorDesign::new(&problem.design, problem.rows, problem.cols)
        .expect("column-major design covers every entry");
    form_xtx(layout.x(), layout.nrows(), layout.ncols(), gram);
}

fn reference_dense_matrix(problem: &DenseProblem) -> Mat<f64> {
    let mut gram = vec![0.0; problem.cols * problem.cols];
    fill_reference_dense(problem, &mut gram);
    let cols = problem.cols;
    Mat::from_fn(cols, cols, |row, col| gram[row * cols + col])
}

fn faer_dense_gram(matrix: &Mat<f64>, par: Par, gram: &mut Mat<f64>) {
    matmul(gram, Accum::Replace, matrix.transpose(), matrix, 1.0, par);
}

fn reference_columns(entries: &[Triplet<usize, usize, f64>], ncols: usize) -> Vec<ReferenceColumn> {
    let mut columns = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        columns.push(ReferenceColumn { rows: Vec::new(), values: Vec::new() });
    }
    for entry in entries {
        columns[entry.col].rows.push(entry.row);
        columns[entry.col].values.push(entry.val);
    }
    columns
}

fn sparse_problem(rows: usize, cols: usize, nonzeros_per_column: usize) -> SparseProblem {
    let mut entries = Vec::with_capacity(cols * nonzeros_per_column);
    for col in 0..cols {
        for offset in 0..nonzeros_per_column {
            let row = col + offset;
            let value = ((row + 3 * col) as f64 * 0.01).sin();
            entries.push(Triplet::new(row, col, value));
        }
    }
    let columns = reference_columns(&entries, cols);
    let transpose_entries: Vec<_> =
        entries.iter().map(|entry| Triplet::new(entry.col, entry.row, entry.val)).collect();
    let transpose_columns = reference_columns(&transpose_entries, rows);
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(rows, cols, &entries)
        .expect("sparse design fits");
    let transpose =
        SparseColMat::<usize, f64>::try_new_from_triplets(cols, rows, &transpose_entries)
            .expect("sparse transpose fits");
    let (symbolic, info) = sparse_sparse_matmul_symbolic(transpose.symbolic(), matrix.symbolic())
        .expect("sparse gram pattern fits");
    SparseProblem {
        cols,
        columns,
        transpose_columns,
        factors: SparseFactors { matrix, transpose, symbolic, info },
    }
}

fn fill_reference_sparse(problem: &SparseProblem, gram: &mut [f64]) {
    let cols = problem.cols;
    gram[..cols * cols].fill(0.0);
    for (col, column) in problem.columns.iter().enumerate() {
        for (&row_k, &x_kj) in column.rows.iter().zip(&column.values) {
            let left = &problem.transpose_columns[row_k];
            for (&row_i, &xt_ik) in left.rows.iter().zip(&left.values) {
                gram[row_i * cols + col] += xt_ik * x_kj;
            }
        }
    }
}

fn reference_sparse_matrix(problem: &SparseProblem) -> Mat<f64> {
    let mut gram = vec![0.0; problem.cols * problem.cols];
    fill_reference_sparse(problem, &mut gram);
    let cols = problem.cols;
    Mat::from_fn(cols, cols, |row, col| gram[row * cols + col])
}

fn faer_sparse_gram(factors: &SparseFactors, values: &mut [f64], stack: &mut MemBuffer, par: Par) {
    sparse_sparse_matmul_numeric(
        SparseColMatMut::new(factors.symbolic.as_ref(), values),
        Accum::Replace,
        factors.transpose.as_ref(),
        factors.matrix.as_ref(),
        1.0,
        &factors.info,
        par,
        MemStack::new(stack),
    );
}

fn faer_sparse_matrix(problem: &SparseProblem, scope: FaerScope) -> Mat<f64> {
    let mut values = vec![0.0; problem.factors.symbolic.row_idx().len()];
    let required = sparse_sparse_matmul_numeric_scratch::<usize, f64>(
        problem.factors.symbolic.as_ref(),
        scope.parallelism(),
    );
    let mut stack = scope.scratch_buffer(required).expect("scratch budget admits the product");
    faer_sparse_gram(&problem.factors, &mut values, &mut stack, scope.parallelism());
    SparseColMatRef::new(problem.factors.symbolic.as_ref(), &values).to_dense()
}

fn assert_close(left: &Mat<f64>, right: &Mat<f64>, label: &str) {
    assert_eq!(left.nrows(), right.nrows(), "{label} rows");
    assert_eq!(left.ncols(), right.ncols(), "{label} cols");
    let mut gap = 0.0_f64;
    let mut scale = 0.0_f64;
    for col in 0..left.ncols() {
        for row in 0..left.nrows() {
            let a = left[(row, col)];
            let b = right[(row, col)];
            gap = gap.max((a - b).abs());
            scale = scale.max(a.abs()).max(b.abs());
        }
    }
    let limit = SOLUTION_TOLERANCE * (1.0 + scale);
    assert!(gap <= limit, "{label} differs by {gap}, limit {limit}");
}

fn assert_matches_reference(reference: &Mat<f64>, faer_serial: &Mat<f64>, shape: &str) {
    let mut scale = 0.0_f64;
    for col in 0..reference.ncols() {
        for row in 0..reference.nrows() {
            scale = scale.max(reference[(row, col)].abs());
        }
    }
    assert!(scale > 0.0, "{shape} reference gram is all zeros");
    assert_close(reference, faer_serial, &format!("{shape} faer versus reference"));
}

fn check_worker_agreement(shape: &str, serial: &Mat<f64>, parallel: &Mat<f64>, workers: u32) {
    let label = format!("{shape} serial versus {workers} workers");
    assert_close(serial, parallel, &label);
}

fn check_dense(problem: &DenseProblem, shape: &str) {
    let reference = reference_dense_matrix(problem);
    let mut serial_gram = Mat::<f64>::zeros(problem.cols, problem.cols);
    faer_dense_gram(&problem.matrix, serial_scope().parallelism(), &mut serial_gram);
    assert_matches_reference(&reference, &serial_gram, shape);
    let mut parallel_gram = Mat::<f64>::zeros(problem.cols, problem.cols);
    for workers in [4, 16] {
        faer_dense_gram(&problem.matrix, faer_scope(workers).parallelism(), &mut parallel_gram);
        check_worker_agreement(shape, &serial_gram, &parallel_gram, workers);
    }
}

fn check_sparse(problem: &SparseProblem, shape: &str) {
    let reference = reference_sparse_matrix(problem);
    let serial_gram = faer_sparse_matrix(problem, serial_scope());
    assert_matches_reference(&reference, &serial_gram, shape);
    for workers in [4, 16] {
        let parallel_gram = faer_sparse_matrix(problem, faer_scope(workers));
        check_worker_agreement(shape, &serial_gram, &parallel_gram, workers);
    }
}

fn bench_reference_dense(c: &mut Criterion, name: &str, problem: &DenseProblem) {
    let mut gram = vec![0.0; problem.cols * problem.cols];
    c.bench_function(name, |b| {
        b.iter(|| {
            fill_reference_dense(black_box(problem), black_box(&mut gram));
            black_box(gram[0]);
        });
    });
}

fn bench_faer_dense(c: &mut Criterion, name: &str, problem: &DenseProblem, scope: FaerScope) {
    let par = scope.parallelism();
    let mut gram = Mat::<f64>::zeros(problem.cols, problem.cols);
    c.bench_function(name, |b| {
        b.iter(|| {
            faer_dense_gram(black_box(&problem.matrix), par, black_box(&mut gram));
            black_box(gram[(0, 0)]);
        });
    });
}

fn bench_reference_sparse(c: &mut Criterion, name: &str, problem: &SparseProblem) {
    let mut gram = vec![0.0; problem.cols * problem.cols];
    c.bench_function(name, |b| {
        b.iter(|| {
            fill_reference_sparse(black_box(problem), black_box(&mut gram));
            black_box(gram[0]);
        });
    });
}

fn bench_faer_sparse(c: &mut Criterion, name: &str, problem: &SparseProblem, scope: FaerScope) {
    let par = scope.parallelism();
    let mut values = vec![0.0; problem.factors.symbolic.row_idx().len()];
    let required =
        sparse_sparse_matmul_numeric_scratch::<usize, f64>(problem.factors.symbolic.as_ref(), par);
    let mut stack = scope.scratch_buffer(required).expect("scratch budget admits the product");
    c.bench_function(name, |b| {
        b.iter(|| {
            faer_sparse_gram(
                black_box(&problem.factors),
                black_box(&mut values),
                black_box(&mut stack),
                par,
            );
            black_box(values[0]);
        });
    });
}

fn bench_dense_layouts(c: &mut Criterion) {
    let small = dense_problem(SMALL_ROWS, SMALL_COLS);
    let large = dense_problem(LARGE_ROWS, LARGE_COLS);
    check_dense(&small, "dense small");
    check_dense(&large, "dense large");

    bench_reference_dense(c, "reference_dense_gram_small", &small);
    bench_reference_dense(c, "reference_dense_gram_large", &large);
    bench_faer_dense(c, "faer_dense_gram_small_serial", &small, serial_scope());
    bench_faer_dense(c, "faer_dense_gram_small_threads_4", &small, faer_scope(4));
    bench_faer_dense(c, "faer_dense_gram_small_threads_16", &small, faer_scope(16));
    bench_faer_dense(c, "faer_dense_gram_large_serial", &large, serial_scope());
    bench_faer_dense(c, "faer_dense_gram_large_threads_4", &large, faer_scope(4));
    bench_faer_dense(c, "faer_dense_gram_large_threads_16", &large, faer_scope(16));
}

fn bench_sparse_layouts(c: &mut Criterion) {
    let small = sparse_problem(SMALL_SPARSE_ROWS, SMALL_SPARSE_COLS, SMALL_NONZEROS_PER_COLUMN);
    let large = sparse_problem(LARGE_SPARSE_ROWS, LARGE_SPARSE_COLS, LARGE_NONZEROS_PER_COLUMN);
    check_sparse(&small, "sparse small");
    check_sparse(&large, "sparse large");

    bench_reference_sparse(c, "reference_sparse_gram_small", &small);
    bench_reference_sparse(c, "reference_sparse_gram_large", &large);
    bench_faer_sparse(c, "faer_sparse_gram_small_serial", &small, serial_scope());
    bench_faer_sparse(c, "faer_sparse_gram_small_threads_4", &small, faer_scope(4));
    bench_faer_sparse(c, "faer_sparse_gram_small_threads_16", &small, faer_scope(16));
    bench_faer_sparse(c, "faer_sparse_gram_large_serial", &large, serial_scope());
    bench_faer_sparse(c, "faer_sparse_gram_large_threads_4", &large, faer_scope(4));
    bench_faer_sparse(c, "faer_sparse_gram_large_threads_16", &large, faer_scope(16));
}

fn bench_blas_layout(c: &mut Criterion) {
    bench_dense_layouts(c);
    bench_sparse_layouts(c);
}

criterion_group!(benches, bench_blas_layout);
criterion_main!(benches);
