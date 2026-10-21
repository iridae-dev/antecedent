//! Portable-layout measurement of the four N15 kernels.
//!
//! This is not the mixed `reductions` or `linalg` bench. Each function times one
//! kernel on borrowed vector/bitmap views or `ColumnMajorDesign`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, reason = "benchmark crate root: the measurement prints its protocol")]

use std::hint::black_box;
use std::time::Instant;

use antecedent_kernels::{BitMaskView, F64VectorView, portable, scalar};
use antecedent_stats::gram::{
    ColumnMajorDesign, DesignRow, accumulate_xtx_xty_row, form_xtx, form_xty,
};

const FUSED_N: usize = 65_536;
const GRAM_ROWS: usize = 4_096;
const GRAM_COLS: usize = 32;
const TRIALS: usize = 31;
const WARMUP: usize = 5;

macro_rules! median_ns {
    ($setup:expr, $run:expr) => {{
        for _ in 0..WARMUP {
            $setup;
            $run;
        }
        let mut samples = [0u64; TRIALS];
        for sample in &mut samples {
            $setup;
            let start = Instant::now();
            $run;
            *sample = u64::try_from(start.elapsed().as_nanos()).expect("duration");
        }
        samples.sort_unstable();
        samples[TRIALS / 2]
    }};
}

fn column(n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|i| f64::from(u32::try_from(i).expect("row")) * scale).collect()
}

fn report(kernel: &str, symbol: &str, detail: &str, ns: u64) {
    println!("n15 {kernel} {symbol} {detail} median_ns={ns}");
}

#[test]
fn fused_reductions() {
    let x = column(FUSED_N, 0.01);
    let y = column(FUSED_N, 0.02);
    let xv = F64VectorView::contiguous(&x);
    let yv = F64VectorView::contiguous(&y);
    let n = format!("n={FUSED_N}");
    report(
        "fused_reductions",
        "masked_sum",
        &n,
        median_ns!(black_box(()), black_box(portable::masked_sum(black_box(xv), None).unwrap())),
    );
    report(
        "fused_reductions",
        "masked_mean",
        &n,
        median_ns!(black_box(()), black_box(portable::masked_mean(black_box(xv), None).unwrap())),
    );
    report(
        "fused_reductions",
        "masked_variance",
        &n,
        median_ns!(
            black_box(()),
            black_box(portable::masked_variance(black_box(xv), None).unwrap())
        ),
    );
    report(
        "fused_reductions",
        "masked_covariance",
        &n,
        median_ns!(
            black_box(()),
            black_box(portable::masked_covariance(black_box(xv), black_box(yv), None).unwrap())
        ),
    );
}

#[test]
fn bitmap_mask() {
    let x = column(FUSED_N, 0.01);
    let y = column(FUSED_N, 0.02);
    let bytes = vec![0b0101_0101; FUSED_N.div_ceil(8)];
    let xv = F64VectorView::contiguous(&x);
    let yv = F64VectorView::contiguous(&y);
    let mask = BitMaskView::new(&bytes, x.len()).unwrap();
    let n = format!("n={FUSED_N} bits=alternating");
    report(
        "bitmap_mask",
        "masked_sum",
        &n,
        median_ns!(
            black_box(()),
            black_box(scalar::masked_sum(black_box(xv), Some(black_box(mask))).unwrap())
        ),
    );
    report(
        "bitmap_mask",
        "masked_mean",
        &n,
        median_ns!(
            black_box(()),
            black_box(scalar::masked_mean(black_box(xv), Some(black_box(mask))).unwrap())
        ),
    );
    report(
        "bitmap_mask",
        "masked_variance",
        &n,
        median_ns!(
            black_box(()),
            black_box(scalar::masked_variance(black_box(xv), Some(black_box(mask))).unwrap())
        ),
    );
    report(
        "bitmap_mask",
        "masked_covariance",
        &n,
        median_ns!(
            black_box(()),
            black_box(
                scalar::masked_covariance(black_box(xv), black_box(yv), Some(black_box(mask)))
                    .unwrap(),
            )
        ),
    );
}

#[test]
fn score_gram() {
    let x = column(GRAM_ROWS * GRAM_COLS, 0.01);
    let y = column(GRAM_ROWS, 0.02);
    let design = ColumnMajorDesign::new(&x, GRAM_ROWS, GRAM_COLS).unwrap();
    let mut xtx = vec![0.0; GRAM_COLS * GRAM_COLS];
    let mut xty = vec![0.0; GRAM_COLS];
    let shape = format!("nrows={GRAM_ROWS} ncols={GRAM_COLS}");
    report(
        "score_gram",
        "form_xtx",
        &shape,
        median_ns!(black_box(()), {
            form_xtx(black_box(design.x()), design.nrows(), design.ncols(), black_box(&mut xtx));
            black_box(xtx[0]);
        }),
    );
    report(
        "score_gram",
        "form_xty",
        &shape,
        median_ns!(black_box(()), {
            form_xty(
                black_box(design.x()),
                design.nrows(),
                design.ncols(),
                black_box(&y),
                black_box(&mut xty),
            );
            black_box(xty[0]);
        }),
    );
    let mut row_store = vec![0.0; GRAM_ROWS * GRAM_COLS];
    for row in 0..GRAM_ROWS {
        for col in 0..GRAM_COLS {
            row_store[row * GRAM_COLS + col] = x[col * GRAM_ROWS + row];
        }
    }
    report(
        "score_gram",
        "accumulate_xtx_xty_row",
        &shape,
        median_ns!(
            {
                xtx.fill(0.0);
                xty.fill(0.0);
            },
            {
                for (design_values, &outcome) in row_store.chunks_exact(GRAM_COLS).zip(y.iter()) {
                    let design_row = DesignRow::new(design_values);
                    accumulate_xtx_xty_row(
                        black_box(design_row.as_slice()),
                        black_box(outcome),
                        black_box(&mut xtx),
                        black_box(&mut xty),
                    );
                }
                black_box(xtx[0]);
            }
        ),
    );
}

#[test]
fn column_scaling() {
    let source = column(FUSED_N, 0.01);
    let mut values = source.clone();
    let n = format!("n={FUSED_N}");
    report(
        "column_scaling",
        "standardize_inplace",
        &n,
        median_ns!(values.copy_from_slice(&source), {
            black_box(portable::standardize_inplace(black_box(&mut values), 1e-12));
        }),
    );
}
