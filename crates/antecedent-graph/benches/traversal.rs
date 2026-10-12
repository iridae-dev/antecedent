//! Graph traversal benchmark baseline .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_graph::{Dag, DenseNodeId, GraphWorkspace};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn chain_dag(n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    g
}

fn bench_traversal(c: &mut Criterion) {
    let g = chain_dag(5_000);
    let mut ws = GraphWorkspace::default();
    c.bench_function("dag_reach_chain_5k", |b| {
        b.iter(|| {
            let ok = g.reaches_with(
                black_box(DenseNodeId::from_raw(0)),
                black_box(DenseNodeId::from_raw(4_999)),
                black_box(&mut ws),
            );
            assert!(ok);
        });
    });
}

/// A lattice of `layers` fully connected 3-node layers hanging off node 0 that never reaches
/// the target, beside the one real path 0 -> target.
fn dead_end_lattice(layers: u32) -> (Dag, DenseNodeId, DenseNodeId) {
    let target = 1 + 3 * layers;
    let mut g = Dag::with_variables(target + 1);
    let id = DenseNodeId::from_raw;
    g.insert_directed(id(0), id(target)).unwrap();
    for j in 0..3 {
        g.insert_directed(id(0), id(1 + j)).unwrap();
    }
    for l in 0..layers - 1 {
        for a in 0..3 {
            for b in 0..3 {
                g.insert_directed(id(1 + 3 * l + a), id(1 + 3 * (l + 1) + b)).unwrap();
            }
        }
    }
    (g, id(0), id(target))
}

fn bench_directed_paths(c: &mut Criterion) {
    let (g, from, to) = dead_end_lattice(9);
    c.bench_function("dag_directed_paths_dead_end_lattice_9", |b| {
        b.iter(|| {
            let (paths, truncated) =
                g.directed_paths_with_budget(black_box(from), black_box(to), 64, 64).unwrap();
            assert_eq!(paths.len(), 1);
            assert!(!truncated);
        });
    });
    let g = chain_dag(200);
    c.bench_function("dag_directed_paths_chain_200", |b| {
        b.iter(|| {
            let (paths, _) = g
                .directed_paths_with_budget(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(199)),
                    64,
                    256,
                )
                .unwrap();
            assert_eq!(paths.len(), 1);
        });
    });
}

criterion_group!(benches, bench_traversal, bench_directed_paths);
criterion_main!(benches);
