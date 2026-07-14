//! Graph traversal benchmark baseline .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_graph::{Dag, DenseNodeId, GraphWorkspace};
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

criterion_group!(benches, bench_traversal);
criterion_main!(benches);
