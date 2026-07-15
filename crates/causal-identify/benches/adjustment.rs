//! Adjustment-set enumeration benchmark .
#![allow(missing_docs, clippy::cast_possible_truncation)]

use causal_core::{AverageEffectQuery, CausalQuery, VariableId};
use causal_graph::{Dag, DenseNodeId};
use causal_identify::{BackdoorIdentifier, IdentificationWorkspace};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn confounded_chain(n_cov: u32) -> Dag {
    // T=0, Y=1, Z_i = 2..
    let mut g = Dag::with_variables(2 + n_cov);
    let t = DenseNodeId::from_raw(0);
    let y = DenseNodeId::from_raw(1);
    g.insert_directed(t, y).unwrap();
    for i in 0..n_cov {
        let z = DenseNodeId::from_raw(2 + i);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
    }
    g
}

fn bench_adjustment(c: &mut Criterion) {
    let g = confounded_chain(8);
    let id = BackdoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    c.bench_function("backdoor_minimal_n8_cov", |b| {
        b.iter(|| {
            let res = id.identify(black_box(&prep), black_box(&q), &mut ws).unwrap();
            assert!(!res.estimands.is_empty());
        });
    });
}

criterion_group!(benches, bench_adjustment);
criterion_main!(benches);
