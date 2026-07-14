//! Orientation local-delta vs naive global rescan .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_core::{Lag, VariableId};
use causal_discovery::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationQueue, OrientationRule,
    OrientationState, RuleDelta, run_orientation_to_fixed_point,
};
use causal_graph::{DenseNodeId, TemporalCpdag};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn chain_cpdag(n: usize) -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        nodes.push(g.add_lagged(VariableId::from_raw(i as u32), Lag::CONTEMPORANEOUS).unwrap());
    }
    // a → b — c — d … with first edge directed to seed Meek R1.
    if n >= 2 {
        g.insert_directed(nodes[0], nodes[1]).unwrap();
    }
    for w in nodes.windows(2).skip(1) {
        g.insert_undirected(w[0], w[1]).unwrap();
    }
    g
}

/// Naive baseline: re-seed the full node set into the queue every rule application.
fn run_orientation_global_rescan(
    graph: &mut TemporalCpdag,
    rules: &[&dyn OrientationRule],
    state: &mut OrientationState,
) -> RuleDelta {
    let mut total = RuleDelta::default();
    for _ in 0..64 {
        let mut any = false;
        for rule in rules {
            let mut queue = OrientationQueue::new();
            for i in 0..graph.node_count() {
                queue.push(DenseNodeId::from_raw(i as u32));
            }
            let d = rule.apply(graph, state, &mut queue).unwrap();
            total.edges_changed += d.edges_changed;
            total.enqueued += d.enqueued;
            if d.edges_changed > 0 {
                any = true;
            }
        }
        if !any {
            total.fixed_point = true;
            break;
        }
    }
    total
}

fn bench_orientation(c: &mut Criterion) {
    let rules: [&dyn OrientationRule; 5] = [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
    let mut group = c.benchmark_group("orientation");
    for n in [16usize, 64, 128] {
        group.bench_function(format!("local_delta_n{n}"), |b| {
            b.iter(|| {
                let mut g = chain_cpdag(n);
                let mut state = OrientationState::default();
                let _ = black_box(run_orientation_to_fixed_point(
                    black_box(&mut g),
                    black_box(&rules),
                    black_box(&mut state),
                ));
            });
        });
        group.bench_function(format!("global_rescan_n{n}"), |b| {
            b.iter(|| {
                let mut g = chain_cpdag(n);
                let mut state = OrientationState::default();
                let _ = black_box(run_orientation_global_rescan(
                    black_box(&mut g),
                    black_box(&rules),
                    black_box(&mut state),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_orientation);
criterion_main!(benches);
