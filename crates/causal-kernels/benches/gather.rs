//! Criterion benchmark for sample gather (Phase 0 baseline workload).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use causal_core::KernelPolicy;
use causal_kernels::{F64VectorView, gather};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_gather(c: &mut Criterion) {
    let n = 100_000usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let src = F64VectorView::contiguous(&data);
    let indices: Vec<usize> = (0..n).step_by(10).collect();
    let mut out = vec![0.0; indices.len()];
    let policy = KernelPolicy::default_policy();

    c.bench_function("gather_stride10_n100k", |b| {
        b.iter(|| {
            gather(black_box(&policy), black_box(src), black_box(&indices), black_box(&mut out));
        });
    });
}

criterion_group!(benches, bench_gather);
criterion_main!(benches);
