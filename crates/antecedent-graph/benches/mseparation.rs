//! ADMG/PAG m-separation sparse and stress benchmarks .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, Pag};
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
    // Diameter of the sparse chain is 100 nodes; budget must cover it or the
    // definite-status search correctly returns SearchBudgetExhausted.
    c.bench_function("msep_pag_sparse_100", |b| {
        b.iter(|| {
            let _ = pag_s
                .is_m_separated(
                    black_box(DenseNodeId::from_raw(0)),
                    black_box(DenseNodeId::from_raw(99)),
                    black_box(&[]),
                    64,
                    128,
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
                    128,
                    128,
                )
                .unwrap();
        });
    });
}

/// x - {a, b, c} - y with a - b - c, every edge circle-circle: 16 circle endpoints, the
/// audit ceiling, and a conditioning set the marks alone cannot settle.
fn circle_diamond() -> (Pag, [DenseNodeId; 3], DenseNodeId, DenseNodeId) {
    let n = DenseNodeId::from_raw;
    let mut g = Pag::with_variables(5);
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 4), (2, 4), (3, 4)] {
        g.insert_circle_circle(n(a), n(b)).unwrap();
    }
    (g, [n(1), n(2), n(3)], n(0), n(4))
}

/// `layers` layers of three nodes, each node a parent of every node in the next layer.
fn directed_lattice(layers: u32) -> (Pag, DenseNodeId, DenseNodeId) {
    let n = DenseNodeId::from_raw;
    let total = 2 + 3 * layers;
    let mut g = Pag::with_variables(total);
    for j in 0..3 {
        g.insert_directed(n(0), n(1 + j)).unwrap();
        g.insert_directed(n(1 + 3 * (layers - 1) + j), n(total - 1)).unwrap();
    }
    for l in 0..layers - 1 {
        for a in 0..3 {
            for b in 0..3 {
                g.insert_directed(n(1 + 3 * l + a), n(1 + 3 * (l + 1) + b)).unwrap();
            }
        }
    }
    (g, n(0), n(total - 1))
}

fn bench_pag_class(c: &mut Criterion) {
    let (g, z, x, y) = circle_diamond();
    c.bench_function("msep_pag_class_fallback_circle_diamond", |b| {
        b.iter(|| {
            let status =
                g.m_separation_status(black_box(x), black_box(y), black_box(&z), 256, 8).unwrap();
            black_box(status);
        });
    });
    let (g, x, y) = directed_lattice(4);
    let z = [DenseNodeId::from_raw(4), DenseNodeId::from_raw(5), DenseNodeId::from_raw(6)];
    c.bench_function("msep_pag_walk_directed_lattice_4", |b| {
        b.iter(|| {
            let search = g.definite_status_paths(black_box(x), black_box(y), 1 << 20, 14).unwrap();
            black_box(&search);
            let _ = g.m_separation_status(black_box(x), black_box(y), black_box(&z), 1 << 20, 14);
        });
    });
}

criterion_group!(benches, bench_msep, bench_pag_class);
criterion_main!(benches);
