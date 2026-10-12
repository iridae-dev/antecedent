//! Temporal unfold and materialize benchmark.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_core::{Lag, TemporalIndexer, VariableId};
use antecedent_graph::TemporalDag;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn template(n: u32) -> TemporalDag {
    let mut g = TemporalDag::empty();
    let count = usize::try_from(n).unwrap_or(0);
    let mut past = Vec::with_capacity(count);
    let mut now = Vec::with_capacity(count);
    for i in 0..n {
        past.push(g.add_lagged(VariableId::from_raw(i), Lag::from_raw(1)).unwrap());
        now.push(g.add_lagged(VariableId::from_raw(i), Lag::CONTEMPORANEOUS).unwrap());
    }
    for (i, &past_node) in past.iter().enumerate() {
        g.insert_directed(past_node, now[i]).unwrap();
        if let Some(&next) = now.get(i + 1) {
            g.insert_directed(now[i], next).unwrap();
        }
    }
    g
}

fn bench_unfold(c: &mut Criterion) {
    let g = template(6);
    let indexer = TemporalIndexer::new(6, 1, 4).unwrap();
    c.bench_function("temporal_unfold_materialize_6vars", |b| {
        b.iter(|| {
            let unfolded = g.unfold(black_box(indexer.clone())).unwrap();
            black_box(unfolded.dag.node_count());
        });
    });
}

criterion_group!(benches, bench_unfold);
criterion_main!(benches);
