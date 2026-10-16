//! Criterion benchmark for partial-correlation batches .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_kernels::{ParCorrMode, ParCorrQuery, ParCorrWorkspace, partial_correlation_batch};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// PC-style batch: X and Y fixed, many conditioning subsets of a shared pool, so queries
/// repeat conditioning columns and (column, target) products.
fn bench_parcorr_shared(c: &mut Criterion) {
    let n = 4_000usize;
    let p = 14usize;
    let cols_owned: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| ((i * (j + 3)) as f64 * 0.37).sin() + (j as f64) * 0.01).collect())
        .collect();
    let col_refs: Vec<&[f64]> = cols_owned.iter().map(Vec::as_slice).collect();
    let mut queries = Vec::new();
    let mut z_flat = Vec::new();
    for a in 2..p {
        for b in (a + 1)..p {
            for d in [None, Some((a + b) % (p - 2) + 2)] {
                let z_start = z_flat.len();
                z_flat.extend([a, b]);
                if let Some(d) = d.filter(|&d| d != a && d != b) {
                    z_flat.push(d);
                }
                queries.push(ParCorrQuery { x: 0, y: 1, z_start, z_len: z_flat.len() - z_start });
            }
        }
    }
    let mut out = vec![None; queries.len()];
    let mut ws = ParCorrWorkspace::default();
    for (name, mode) in [("portable", ParCorrMode::Portable), ("native", ParCorrMode::Native)] {
        c.bench_function(&format!("parcorr_shared_z_{name}_n4k_p14_q{}", queries.len()), |b| {
            b.iter(|| {
                partial_correlation_batch(
                    black_box(&col_refs),
                    black_box(&queries),
                    black_box(&z_flat),
                    black_box(&mut out),
                    black_box(&mut ws),
                    mode,
                )
                .unwrap();
            });
        });
    }
}

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
                ParCorrMode::Portable,
            )
            .unwrap();
        });
    });
}

criterion_group!(benches, bench_parcorr, bench_parcorr_shared);
criterion_main!(benches);
