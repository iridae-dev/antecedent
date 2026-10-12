//! Adjustment-set enumeration benchmark .
#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_identify::{BackdoorIdentifier, IdIdentifier, IdentificationWorkspace};
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

/// Subset search inside the default cap of 40. A pool larger than the cap takes
/// the O-set shortcut instead of enumerating.
fn bench_enumeration(c: &mut Criterion) {
    let g = confounded_chain(12);
    let id = BackdoorIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    c.bench_function("backdoor_subset_search_n12_under_cap40", |b| {
        b.iter(|| {
            let res = id.identify(black_box(&prep), black_box(&q), &mut ws).unwrap();
            assert!(!res.estimands.is_empty());
        });
    });
}

fn chain_admg(n: u32) -> Admg {
    let mut g = Admg::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    g
}

fn bench_id_recursion(c: &mut Criterion) {
    let g = chain_admg(12);
    let id = IdIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(11),
    ));
    c.bench_function("id_recursion_chain_12", |b| {
        b.iter(|| {
            let res = id.identify(black_box(&prep), black_box(&q), &mut ws).unwrap();
            assert!(!res.estimands.is_empty());
        });
    });
}

criterion_group!(benches, bench_adjustment, bench_enumeration, bench_id_recursion);
criterion_main!(benches);
