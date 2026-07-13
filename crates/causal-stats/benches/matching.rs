//! Matching-index benchmark (Phase 4).
#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use causal_stats::{MatchingDistance, MatchingIndex};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_matching(c: &mut Criterion) {
    let n = 500usize;
    let dim = 4usize;
    let mut donors = vec![0.0; n * dim];
    let mut queries = vec![0.0; n * dim];
    let mut donor_rows = Vec::with_capacity(n);
    for i in 0..n {
        donor_rows.push(i);
        for d in 0..dim {
            donors[i * dim + d] = (i * dim + d) as f64 * 0.001;
            queries[i * dim + d] = donors[i * dim + d] + 0.01;
        }
    }
    let idx = MatchingIndex::exact(&donors, dim, &donor_rows, MatchingDistance::Euclidean).unwrap();
    let mut out_row = vec![0usize; n];
    let mut out_d = vec![0.0; n];
    c.bench_function("matching_exact_n500_d4", |b| {
        b.iter(|| {
            let m = idx
                .match_all(black_box(&queries), n, None, &mut out_row, &mut out_d)
                .unwrap();
            assert_eq!(m, n as u32);
        });
    });
}

criterion_group!(benches, bench_matching);
criterion_main!(benches);
