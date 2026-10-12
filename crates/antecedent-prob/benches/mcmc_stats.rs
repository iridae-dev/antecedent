//! Rank-normalized R-hat / Geyer ESS benchmark.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use antecedent_prob::mcmc_summary;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn fill_ar1(n_chains: usize, n_draws: usize, rho: f64, seed: u64) -> Vec<f64> {
    let n = n_chains * n_draws;
    let mut rng = antecedent_core::CausalRng::from_seed(seed);
    let innov: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let mut samples = vec![0.0; n];
    let scale = (1.0 - rho * rho).sqrt();
    for c in 0..n_chains {
        let base = c * n_draws;
        samples[base] = innov[base];
        for d in 1..n_draws {
            samples[base + d] = rho * samples[base + d - 1] + scale * innov[base + d];
        }
    }
    samples
}

fn bench_mcmc_stats(c: &mut Criterion) {
    let n_chains = 4usize;
    let n_draws = 256usize;
    let n_params = 2usize;
    let mut samples = vec![0.0; n_chains * n_draws * n_params];
    for p in 0..n_params {
        let col = fill_ar1(n_chains, n_draws, 0.5, 11 + p as u64);
        // Column-major param: samples[(chain * n_draws + draw) * n_params + param]
        for (i, v) in col.iter().enumerate() {
            samples[i * n_params + p] = *v;
        }
    }

    c.bench_function("mcmc_summary_c4_n256_p2", |b| {
        b.iter(|| {
            black_box(mcmc_summary(
                black_box(&samples),
                black_box(n_chains),
                black_box(n_draws),
                black_box(n_params),
            ));
        });
    });
}

criterion_group!(benches, bench_mcmc_stats);
criterion_main!(benches);
