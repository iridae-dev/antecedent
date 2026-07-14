//! Criterion benchmark for partial-correlation batches .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use causal_kernels::{ParCorrQuery, ParCorrWorkspace, partial_correlation_batch};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_parcorr(c: &mut Criterion) {
    let n = 2_000usize;
    let p = 8usize;
    let mut cols_owned: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        cols_owned.push((0..n).map(|i| ((i + j * 17) as f64).sin()).collect());
    }
    let col_refs: Vec<&[f64]> = cols_owned.iter().map(Vec::as_slice).collect();
    // 64 queries: x=0,y=1 with Z = {2..k} cycling
    let mut queries = Vec::new();
    let mut z_flat = Vec::new();
    for k in 0..64 {
        let z_start = z_flat.len();
        let z_len = 1 + (k % 3);
        for t in 0..z_len {
            z_flat.push(2 + (t % (p - 2)));
        }
        queries.push(ParCorrQuery { x: 0, y: 1, z_start, z_len });
    }
    let mut out = vec![None; queries.len()];
    let mut ws = ParCorrWorkspace::default();

    c.bench_function("parcorr_batch64_n2k_p8", |b| {
        b.iter(|| {
            partial_correlation_batch(
                black_box(&col_refs),
                black_box(&queries),
                black_box(&z_flat),
                black_box(&mut out),
                black_box(&mut ws),
                true,
            );
        });
    });
}

criterion_group!(benches, bench_parcorr);
criterion_main!(benches);
