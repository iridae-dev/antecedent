//! ADMG/PAG m-separation sparse and stress benchmarks (Phase 8 exit).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_graph::{Admg, DSeparationWorkspace, DenseNodeId, Pag};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn sparse_admg(n: u32) -> Admg {
    let mut g = Admg::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    for i in (0..n.saturating_sub(2)).step_by(4) {
        let _ = g.insert_bidirected(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 2));
    }
    g
}

fn stress_admg(n: u32) -> Admg {
    let mut g = Admg::with_variables(n);
    for i in 0..n {
        for j in (i + 1)..n.min(i + 5) {
            let _ = g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(j));
        }
        if i + 3 < n {
            let _ = g.insert_bidirected(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 3));
        }
    }
    g
}

fn sparse_pag(n: u32) -> Pag {
    let mut g = Pag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    g
}

fn stress_pag(n: u32) -> Pag {
    let mut g = Pag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        if i % 3 == 0 {
            let _ = g.insert_circle_arrow(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1));
        } else {
            let _ = g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1));
        }
    }
    g
}

fn bench_msep(c: &mut Criterion) {
    let sparse = sparse_admg(200);
    let stress = stress_admg(80);
    let mut ws = DSeparationWorkspace::default();
    let z = [DenseNodeId::from_raw(50)];

    c.bench_function("msep_admg_sparse_200", |b| {
        b.iter(|| {
            let _ = sparse
                .is_m_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(199)),
                    black_box(&z),
                    black_box(&mut ws),
                )
                .unwrap();
        });
    });

    c.bench_function("msep_admg_stress_80", |b| {
        b.iter(|| {
            let _ = stress
                .is_m_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(79)),
                    black_box(&[DenseNodeId::from_raw(10)]),
                    black_box(&mut ws),
                )
                .unwrap();
        });
    });

    let pag_s = sparse_pag(100);
    let pag_t = stress_pag(60);
    c.bench_function("msep_pag_sparse_100", |b| {
        b.iter(|| {
            let _ = pag_s
                .is_m_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(99)),
                    black_box(&[]),
                    32,
                    16,
                )
                .unwrap();
        });
    });

    c.bench_function("msep_pag_stress_60", |b| {
        b.iter(|| {
            let _ = pag_t
                .is_m_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(59)),
                    black_box(&[]),
                    64,
                    16,
                )
                .unwrap();
        });
    });
}

criterion_group!(benches, bench_msep);
criterion_main!(benches);
