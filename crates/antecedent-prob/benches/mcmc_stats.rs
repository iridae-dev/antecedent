//! Rank-normalized R-hat / Geyer ESS benchmark.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::many_single_char_names)]

use antecedent_prob::mcmc_summary;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn fill_ar1(n_chains: usize, n_draws: usize, rho: f64, seed: u64) -> Vec<f64> {
    let n = n_chains * n_draws;
    let mut innov = vec![0.0; n];
    let mut state = seed;
    for v in &mut innov {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = ((state >> 33) as f64) / ((1u64 << 31) as f64);
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = ((state >> 33) as f64) / ((1u64 << 31) as f64);
        let u1 = u1.clamp(1e-12, 1.0 - 1e-12);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
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
