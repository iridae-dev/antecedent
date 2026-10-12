//! Surgical-graph rebuild benchmark.
//!
//! Measures [`Admg::copy_filtered`] the way IDC rebuilds `G` with incoming
//! edges to the intervened nodes removed. The predicate is the measurement;
//! it does not change identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_graph::{Admg, BitSet, CopyLabels, DenseNodeId};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn mixed(n: u32) -> Admg {
    let mut g = Admg::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    for i in (0..n.saturating_sub(3)).step_by(3) {
        let _ = g.insert_bidirected(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 2));
    }
    g
}

fn bench_surgery(c: &mut Criterion) {
    let g = mixed(80);
    let mut intervene = BitSet::with_len(g.node_count());
    intervene.insert(DenseNodeId::from_raw(10));
    intervene.insert(DenseNodeId::from_raw(40));
    c.bench_function("admg_surgery_rebuild_n80", |b| {
        b.iter(|| {
            let rebuilt = g
                .copy_filtered(
                    CopyLabels::Preserve,
                    |from, to| !intervene.contains(from) && !intervene.contains(to),
                    |a, b| !intervene.contains(a) && !intervene.contains(b),
                )
                .unwrap();
            assert_eq!(rebuilt.node_count(), g.node_count());
            black_box(rebuilt);
        });
    });
}

criterion_group!(benches, bench_surgery);
criterion_main!(benches);
