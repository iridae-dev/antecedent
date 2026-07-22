//! Criterion benchmarks for §23.2 reduction kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use causal_core::KernelPolicy;
use causal_kernels::{
    F64VectorView, accumulate_contingency, masked_covariance, pairwise_l1_fill,
    standardize_inplace, weighted_sum,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_reductions(criterion: &mut Criterion) {
    let length = 10_000usize;
    let x: Vec<f64> = (0..length).map(|idx| idx as f64 * 0.01).collect();
    let y: Vec<f64> = (0..length).map(|idx| (idx as f64 * 0.02) + 1.0).collect();
    let w: Vec<f64> = (0..length).map(|idx| 0.5 + (idx % 7) as f64 * 0.1).collect();
    let policy = KernelPolicy::default_policy();
    let xv = F64VectorView::contiguous(&x);
    let yv = F64VectorView::contiguous(&y);

    criterion.bench_function("masked_covariance_n10k", |b| {
        b.iter(|| masked_covariance(black_box(&policy), black_box(xv), black_box(yv), None));
    });

    criterion.bench_function("standardize_inplace_n10k", |b| {
        let mut buf = x.clone();
        b.iter(|| {
            buf.copy_from_slice(&x);
            standardize_inplace(black_box(&policy), black_box(&mut buf), 1e-12)
        });
    });

    criterion.bench_function("weighted_sum_n10k", |b| {
        b.iter(|| weighted_sum(black_box(&policy), black_box(&x), black_box(&w)));
    });

    let pair_len = 256usize;
    let xp: Vec<f64> = (0..pair_len).map(|idx| idx as f64).collect();
    let mut pair_out = vec![0.0; pair_len * pair_len];
    criterion.bench_function("pairwise_l1_n256", |b| {
        b.iter(|| {
            pairwise_l1_fill(black_box(&policy), black_box(&xp), black_box(&mut pair_out));
        });
    });

    let xc: Vec<u32> = (0..length)
        .map(|idx| u32::try_from(idx % 8).unwrap_or(0))
        .collect();
    let yc: Vec<u32> = (0..length)
        .map(|idx| u32::try_from(idx % 5).unwrap_or(0))
        .collect();
    let mut table = vec![0.0; 8 * 5];
    criterion.bench_function("accumulate_contingency_n10k", |b| {
        b.iter(|| {
            table.fill(0.0);
            accumulate_contingency(
                black_box(&policy),
                black_box(&xc),
                black_box(&yc),
                black_box(&mut table),
                5,
            );
        });
    });
}

criterion_group!(benches, bench_reductions);
criterion_main!(benches);
