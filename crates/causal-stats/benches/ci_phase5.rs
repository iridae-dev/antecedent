//! Phase 5 CI batch benches (conditioning size + missingness).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use causal_core::ExecutionContext;
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, KnnCmi, PartialCorrelation,
    SignificanceMethod,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn columns(n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..p)
        .map(|j| {
            (0..n)
                .map(|i| ((i + j * 17) as f64 * 0.01).sin() + 0.05 * (i as f64))
                .collect()
        })
        .collect()
}

/// Simulate missingness by dropping a fraction of leading rows (complete-case).
fn complete_case(cols: &[Vec<f64>], drop: usize) -> Vec<&[f64]> {
    cols.iter().map(|c| &c[drop..]).collect()
}

fn bench_ci_batches(c: &mut Criterion) {
    let n = 400usize;
    let raw = columns(n, 6);
    let ctx = ExecutionContext::for_tests(1);
    let mut group = c.benchmark_group("ci_batch_parcorr");
    for z_len in [0usize, 1, 2, 4] {
        group.bench_function(format!("z{z_len}_full"), |b| {
            let cols = complete_case(&raw, 0);
            let z_flat: Vec<usize> = (2..2 + z_len).collect();
            let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len }];
            let mut ws = CiWorkspace::default();
            b.iter(|| {
                let req = CiBatchRequest {
                    columns: black_box(&cols),
                    queries: black_box(&queries),
                    z_flat: black_box(&z_flat),
                    significance: SignificanceMethod::Analytic,
                };
                let _ = black_box(PartialCorrelation::new().test_batch(&req, &mut ws, &ctx));
            });
        });
        group.bench_function(format!("z{z_len}_missing20"), |b| {
            let drop = n / 5;
            let cols = complete_case(&raw, drop);
            let z_flat: Vec<usize> = (2..2 + z_len).collect();
            let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len }];
            let mut ws = CiWorkspace::default();
            b.iter(|| {
                let req = CiBatchRequest {
                    columns: black_box(&cols),
                    queries: black_box(&queries),
                    z_flat: black_box(&z_flat),
                    significance: SignificanceMethod::Analytic,
                };
                let _ = black_box(PartialCorrelation::new().test_batch(&req, &mut ws, &ctx));
            });
        });
    }
    group.finish();
}

fn bench_knn_reuse(c: &mut Criterion) {
    let n = 120usize;
    let raw = columns(n, 3);
    let cols = complete_case(&raw, 0);
    let z_flat = [2usize];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }; 8];
    let ctx = ExecutionContext::for_tests(2);
    c.bench_function("knn_cmi_reuse_batch8", |b| {
        let mut ws = CiWorkspace::default();
        // Warmup so generation is stable.
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
        };
        let _ = KnnCmi::new(3).test_batch(&req, &mut ws, &ctx);
        let gen0 = ws.knn.index_generation;
        b.iter(|| {
            let req = CiBatchRequest {
                columns: black_box(&cols),
                queries: black_box(&queries),
                z_flat: black_box(&z_flat),
                significance: SignificanceMethod::Analytic,
            };
            let _ = black_box(KnnCmi::new(3).test_batch(&req, &mut ws, &ctx));
            assert_eq!(ws.knn.index_generation, gen0, "kNN must not rebuild index per query batch");
        });
    });
}

criterion_group!(benches, bench_ci_batches, bench_knn_reuse);
criterion_main!(benches);
