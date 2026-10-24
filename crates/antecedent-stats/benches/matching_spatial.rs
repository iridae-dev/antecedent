//! Exact multivariate nearest-neighbour search: the bounding-box tree against the split-plane
//! tree and a linear scan, on uniform and clustered donors across dimensions and sizes.
//!
//! Each case first checks that every index returns the donor and the distance bits of the
//! linear scan for every query, then times the index build and a full query pass, and prints
//! the retained bytes of both trees to stderr. This measures the candidate index on its own;
//! the production matching path keeps `MatchingIndex` and its memory lease until these records
//! support changing it.
//!
//! ```text
//! cargo bench -p antecedent-stats --bench matching_spatial
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, reason = "criterion generates benchmark entry points")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "benchmark scaffolding builds deterministic data from small integers"
)]

use std::time::Duration;

use antecedent_core::ExecutionContext;
use antecedent_stats::{BoxTreeIndex, MatchingDistance, MatchingIndex};
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn uniform(n: usize, dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n * dim).map(|_| rng.next()).collect()
}

fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let centres: Vec<f64> = (0..8 * dim).map(|_| rng.next() * 10.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let centre = ((rng.next() * 8.0) as usize).min(7);
        for d in 0..dim {
            out.push(centres[centre * dim + d] + rng.next() - 0.5);
        }
    }
    out
}

fn scan_nearest(donors: &[f64], dim: usize, query: &[f64]) -> (usize, f64) {
    let mut best = (usize::MAX, f64::INFINITY);
    for (position, row) in donors.chunks_exact(dim).enumerate() {
        let distance = row
            .iter()
            .zip(query)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt();
        if distance < best.1 {
            best = (position, distance);
        }
    }
    best
}

fn bench_spatial(c: &mut Criterion) {
    let ctx = ExecutionContext::for_tests(1);
    let mut group = c.benchmark_group("spatial");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(2));
    for kind in ["uniform", "clustered"] {
        for dim in [2usize, 4, 8, 16] {
            for n in [2_000usize, 20_000, 100_000] {
                let make = if kind == "uniform" { uniform } else { clustered };
                let donors = make(n, dim, 11);
                let n_queries = if n <= 20_000 { 500 } else { 2_000 };
                let queries = make(n_queries, dim, 13);
                let ids: Vec<usize> = (0..n).collect();
                let kd = MatchingIndex::exact_within(
                    &donors,
                    dim,
                    &ids,
                    MatchingDistance::Euclidean,
                    &ctx,
                )
                .unwrap();
                let boxed = BoxTreeIndex::exact_within(&donors, dim, &ctx).unwrap();
                let mut kd_rows = vec![0usize; n_queries];
                let mut kd_dist = vec![0.0; n_queries];
                let mut box_rows = vec![0usize; n_queries];
                let mut box_dist = vec![0.0; n_queries];
                kd.match_all_within(&queries, n_queries, None, &mut kd_rows, &mut kd_dist, &ctx)
                    .unwrap();
                boxed
                    .match_all_within(&queries, n_queries, None, &mut box_rows, &mut box_dist, &ctx)
                    .unwrap();
                assert_eq!(kd_rows, box_rows);
                assert!(kd_dist.iter().zip(&box_dist).all(|(a, b)| a.to_bits() == b.to_bits()));
                if n <= 20_000 {
                    for (q, query) in queries.chunks_exact(dim).enumerate() {
                        let want = scan_nearest(&donors, dim, query);
                        assert_eq!(
                            (box_rows[q], box_dist[q].to_bits()),
                            (want.0, want.1.to_bits())
                        );
                    }
                }
                eprintln!(
                    "spatial/{kind}/n_{n}/d_{dim} retained bytes: split-plane tree {}, \
                     bounding-box tree {} (quote {})",
                    kd.retained_memory_bytes(),
                    boxed.retained_memory_bytes(),
                    BoxTreeIndex::quote_bytes(n, dim).unwrap_or(u64::MAX),
                );
                let id = format!("{kind}/n_{n}/d_{dim}");
                group.bench_function(format!("{id}/split_plane/build"), |b| {
                    b.iter_batched(
                        || (),
                        |()| {
                            black_box(
                                MatchingIndex::exact_within(
                                    black_box(&donors),
                                    dim,
                                    &ids,
                                    MatchingDistance::Euclidean,
                                    &ctx,
                                )
                                .unwrap(),
                            )
                        },
                        BatchSize::LargeInput,
                    );
                });
                group.bench_function(format!("{id}/bounding_box/build"), |b| {
                    b.iter_batched(
                        || (),
                        |()| {
                            black_box(
                                BoxTreeIndex::exact_within(black_box(&donors), dim, &ctx).unwrap(),
                            )
                        },
                        BatchSize::LargeInput,
                    );
                });
                group.bench_function(format!("{id}/split_plane/query"), |b| {
                    b.iter(|| {
                        kd.match_all_within(
                            black_box(&queries),
                            n_queries,
                            None,
                            &mut kd_rows,
                            &mut kd_dist,
                            &ctx,
                        )
                        .unwrap()
                    });
                });
                group.bench_function(format!("{id}/bounding_box/query"), |b| {
                    b.iter(|| {
                        boxed
                            .match_all_within(
                                black_box(&queries),
                                n_queries,
                                None,
                                &mut box_rows,
                                &mut box_dist,
                                &ctx,
                            )
                            .unwrap()
                    });
                });
                if n <= 20_000 {
                    group.bench_function(format!("{id}/linear_scan/query"), |b| {
                        b.iter(|| {
                            let mut checksum = 0usize;
                            for query in black_box(&queries).chunks_exact(dim) {
                                checksum ^= scan_nearest(&donors, dim, query).0;
                            }
                            checksum
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_spatial);
criterion_main!(benches);
