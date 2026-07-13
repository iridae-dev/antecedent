//! Phase 5 CI batch benches (conditioning size + missingness).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use causal_core::ExecutionContext;
use causal_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependence, GSquared, KnnCmi,
    PartialCorrelation, RobustPartialCorrelation, SignificanceMethod,
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

fn discrete_columns(n: usize, p: usize, levels: i32) -> Vec<Vec<f64>> {
    (0..p)
        .map(|j| {
            (0..n)
                .map(|i| ((i + j * 3) % levels as usize) as f64)
                .collect()
        })
        .collect()
}

/// Simulate missingness via a keep-mask (true = observe), then gather complete-case rows.
fn masked_complete_case<'a>(cols: &'a [Vec<f64>], keep: &[bool], scratch: &'a mut Vec<Vec<f64>>) -> Vec<&'a [f64]> {
    scratch.clear();
    for c in cols {
        let mut out = Vec::with_capacity(keep.iter().filter(|k| **k).count());
        for (i, &k) in keep.iter().enumerate() {
            if k {
                out.push(c[i]);
            }
        }
        scratch.push(out);
    }
    scratch.iter().map(Vec::as_slice).collect()
}

fn keep_mask(n: usize, drop_frac: f64, seed: u64) -> Vec<bool> {
    let mut rng = seed;
    let mut keep = vec![true; n];
    let drop_n = ((n as f64) * drop_frac) as usize;
    let mut dropped = 0usize;
    let mut i = 0usize;
    while dropped < drop_n && i < n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng >> 33) % 5 == 0 {
            keep[i] = false;
            dropped += 1;
        }
        i += 1;
    }
    // Force remaining drops from the end if the sparse walk under-dropped.
    let mut j = n;
    while dropped < drop_n && j > 0 {
        j -= 1;
        if keep[j] {
            keep[j] = false;
            dropped += 1;
        }
    }
    keep
}

fn bench_one_ci<C: ConditionalIndependence>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    ci: C,
    raw: &[Vec<f64>],
    z_len: usize,
    keep: Option<&[bool]>,
) {
    let n = raw[0].len();
    let ctx = ExecutionContext::for_tests(1);
    let mut scratch = Vec::new();
    let all_keep = vec![true; n];
    let mask = keep.unwrap_or(&all_keep);
    group.bench_function(name, |b| {
        let mut ws = CiWorkspace::default();
        b.iter(|| {
            let cols = masked_complete_case(raw, mask, &mut scratch);
            let z_flat: Vec<usize> = (2..2 + z_len).collect();
            let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len }];
            let req = CiBatchRequest {
                columns: black_box(&cols),
                queries: black_box(&queries),
                z_flat: black_box(&z_flat),
                significance: SignificanceMethod::Analytic,
            };
            let _ = black_box(ci.test_batch(&req, &mut ws, &ctx));
        });
    });
}

fn bench_ci_batches(c: &mut Criterion) {
    let n = 400usize;
    let raw = columns(n, 6);
    let disc = discrete_columns(n, 4, 4);
    let miss = keep_mask(n, 0.20, 42);

    let mut group = c.benchmark_group("ci_batch_parcorr");
    for z_len in [0usize, 1, 2, 4] {
        bench_one_ci(&mut group, &format!("z{z_len}_full"), PartialCorrelation::new(), &raw, z_len, None);
        bench_one_ci(
            &mut group,
            &format!("z{z_len}_missing20"),
            PartialCorrelation::new(),
            &raw,
            z_len,
            Some(&miss),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("ci_batch_robust");
    bench_one_ci(&mut group, "z1_full", RobustPartialCorrelation::new(), &raw, 1, None);
    bench_one_ci(
        &mut group,
        "z1_missing20",
        RobustPartialCorrelation::new(),
        &raw,
        1,
        Some(&miss),
    );
    group.finish();

    let mut group = c.benchmark_group("ci_batch_gsquared");
    bench_one_ci(&mut group, "z0_full", GSquared::new(), &disc, 0, None);
    bench_one_ci(&mut group, "z1_full", GSquared::new(), &disc, 1, None);
    bench_one_ci(
        &mut group,
        "z1_missing20",
        GSquared::new(),
        &disc,
        1,
        Some(&miss),
    );
    group.finish();

    let mut group = c.benchmark_group("ci_batch_knn");
    let knn_raw = columns(120, 3);
    let knn_miss = keep_mask(120, 0.20, 7);
    bench_one_ci(&mut group, "z1_full", KnnCmi::new(3), &knn_raw, 1, None);
    bench_one_ci(
        &mut group,
        "z1_missing20",
        KnnCmi::new(3),
        &knn_raw,
        1,
        Some(&knn_miss),
    );
    group.finish();
}

fn bench_knn_reuse(c: &mut Criterion) {
    let n = 120usize;
    let raw = columns(n, 3);
    let keep = vec![true; n];
    let mut scratch = Vec::new();
    let cols = masked_complete_case(&raw, &keep, &mut scratch);
    let cols_owned: Vec<Vec<f64>> = cols.iter().map(|c| c.to_vec()).collect();
    let cols_refs: Vec<&[f64]> = cols_owned.iter().map(Vec::as_slice).collect();
    let z_flat = [2usize];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }; 8];
    let ctx = ExecutionContext::for_tests(2);
    c.bench_function("knn_cmi_reuse_batch8", |b| {
        let mut ws = CiWorkspace::default();
        let req = CiBatchRequest {
            columns: &cols_refs,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
        };
        let _ = KnnCmi::new(3).test_batch(&req, &mut ws, &ctx);
        let gen0 = ws.knn.index_generation;
        b.iter(|| {
            let req = CiBatchRequest {
                columns: black_box(&cols_refs),
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
