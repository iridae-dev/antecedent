//! LPCMCI / PAG orientation sparse and stress benchmarks .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_core::{Lag, VariableId};
use causal_discovery::{
    LpcmciDiscriminatingPathRule, LpcmciOrientCollider, LpcmciOrientationRule, LpcmciR1,
    OrientationState, run_lpcmci_orientation,
};
use causal_graph::{DenseNodeId, TemporalPag};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn circle_chain(n: usize) -> TemporalPag {
    let mut g = TemporalPag::empty();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        nodes.push(g.add_lagged(VariableId::from_raw(i as u32), Lag::CONTEMPORANEOUS).unwrap());
    }
    if n >= 2 {
        g.insert_directed(nodes[0], nodes[1]).unwrap();
    }
    for w in nodes.windows(2).skip(1) {
        g.insert_circle_arrow(w[0], w[1]).unwrap();
    }
    g
}

fn bench_pag_orient(c: &mut Criterion) {
    let rules: [&dyn LpcmciOrientationRule; 3] =
        [&LpcmciOrientCollider, &LpcmciR1, &LpcmciDiscriminatingPathRule];

    c.bench_function("pag_orient_sparse_40", |b| {
        b.iter(|| {
            let mut g = circle_chain(40);
            let mut state = OrientationState::default();
            let d = run_lpcmci_orientation(&mut g, &rules, &mut state).unwrap();
            black_box(d);
            black_box(g.node_count());
        });
    });

    c.bench_function("pag_orient_stress_120", |b| {
        b.iter(|| {
            let mut g = circle_chain(120);
            let mut state = OrientationState::default();
            // Seed a few sepsets so collider rule has work.
            for i in 0..g.node_count().saturating_sub(2) {
                let a = DenseNodeId::from_raw(i as u32);
                let c = DenseNodeId::from_raw((i + 2) as u32);
                state.set_sepset(a, c, std::sync::Arc::from([]));
            }
            let d = run_lpcmci_orientation(&mut g, &rules, &mut state).unwrap();
            black_box(d);
        });
    });
}

criterion_group!(benches, bench_pag_orient);
criterion_main!(benches);
