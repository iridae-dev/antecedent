//! d-separation batch benchmark .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_graph::{DSeparationWorkspace, Dag, DenseNodeId};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn sparse_chain(n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    g
}

fn denser_dag(n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    for i in 0..n {
        for j in (i + 1)..n.min(i + 6) {
            let _ = g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(j));
        }
    }
    g
}

fn bench_dsep(c: &mut Criterion) {
    let sparse = sparse_chain(200);
    let dense = denser_dag(80);
    let mut ws = DSeparationWorkspace::default();
    let z_sparse = [DenseNodeId::from_raw(50)];
    let z_dense = [DenseNodeId::from_raw(10), DenseNodeId::from_raw(20)];

    c.bench_function("dsep_sparse_chain_200", |b| {
        b.iter(|| {
            let ok = sparse
                .is_d_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(199)),
                    black_box(&z_sparse),
                    black_box(&mut ws),
                )
                .unwrap();
            assert!(ok);
        });
    });

    c.bench_function("dsep_dense_n80", |b| {
        b.iter(|| {
            let _ = dense
                .is_d_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(79)),
                    black_box(&z_dense),
                    black_box(&mut ws),
                )
                .unwrap();
        });
    });
}

criterion_group!(benches, bench_dsep);
criterion_main!(benches);
